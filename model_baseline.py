import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import math
from transformers import AutoModel

# [핵심] Fourier Projection이 없는 평범한 인코더
class StandardMSEncoder(nn.Module):
    def __init__(self, encoder_config, embedding_dim):
        super().__init__()
        
        d_model = encoder_config['d_model']
        
        # 1. 평범한 선형 임베딩 (No Gaussian Fourier)
        # log(mz) 값 하나를 d_model 크기로 그냥 넙적하게 늘림
        self.mz_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Intensity 임베딩
        self.int_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Fusion (No Concatenation -> Summation)
        # "위치"와 "세기"를 그냥 더해버림 (기존 논문들의 방식)
        # *참고: Concatenation을 안 하므로 입력 차원 변환 불필요
        
        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_config['nhead'],
            dim_feedforward=d_model * 4,
            dropout=encoder_config['dropout'],
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=encoder_config['n_layers']
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x, mask):
        # x: [B, S, 2]
        mz_val = x[:, :, 0:1]
        int_val = x[:, :, 1:2]
        
        # 단순 임베딩
        emb_mz = self.mz_embed(mz_val)
        emb_int = self.int_embed(int_val)
        
        # [핵심 차이] 더하기 (Summation)
        # 물리적 채널을 섞어버림 -> 해상도 저하 유도
        x_emb = emb_mz + emb_int 
        
        # CLS 토큰 추가
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # 마스킹
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)
        transformer_mask = ~full_mask
        
        # 트랜스포머 통과
        out = self.transformer_encoder(x_emb, src_key_padding_mask=transformer_mask)
        
        # 출력
        cls_output = out[:, 0, :]
        projected = self.projection(cls_output)
        return F.normalize(projected, p=2, dim=1)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
            )
        
        if config.TEXT_ENCODER['freeze_bert']:
            for param in self.bert.parameters():
                param.requires_grad = False
                
                
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # We are just getting embeddings
            r=config.LORA['r'],  # Rank (hyperparameter, 8 or 16 is common)
            lora_alpha=config.LORA['lora_alpha'], # (hyperparameter, often 2*r)
            lora_dropout=config.LORA['lora_dropout'],
            target_modules=config.LORA['target_modules'] # Apply to attention layers
        )
        self.bert = get_peft_model(self.bert, lora_config)
        print("\nTextEncoder (LoRA) Trainable Parameters:")
        self.bert.print_trainable_parameters()
        
        bert_dim = self.bert.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean Pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        mean_pooled_embedding = sum_embeddings / (sum_mask + 1e-9)
        
        projected = self.projection(mean_pooled_embedding)
        x = F.normalize(projected, p=2, dim=1)
        return x


class BaselineCLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 여기서 StandardMSEncoder를 사용!
        self.ms_encoder = StandardMSEncoder(config.MS_ENCODER, config.EMBEDDING_DIM)
        self.text_encoder = TextEncoder(
            config.TEXT_ENCODER['model_name'],
            config.EMBEDDING_DIM
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.TEMPERATURE))

    def forward(self, peak_sequence, peak_mask, input_ids, attention_mask):
        # 1. 임베딩 추출
        spec_embeds = self.ms_encoder(peak_sequence, peak_mask)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        # 2. Loss 계산 (이 부분이 누락되었을 가능성이 큽니다)
        batch_size = spec_embeds.shape[0]
        
        # 코사인 유사도 계산 (Logit Scale 적용)
        logits_per_spec = (spec_embeds @ text_embeds.T) * self.logit_scale.exp()
        logits_per_text = logits_per_spec.T
        
        # 정답 라벨 (대각선이 정답: 0, 1, 2, ... Batch-1)
        # 주의: config.DEVICE가 잘 작동하는지 확인하세요. 에러나면 input_ids.device를 써도 됩니다.
        labels = torch.arange(batch_size, device=input_ids.device, dtype=torch.long)
        
        # Symmetric Cross Entropy Loss (CLIP Loss)
        loss_spec = F.cross_entropy(logits_per_spec, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_spec + loss_text) / 2
        
        return loss