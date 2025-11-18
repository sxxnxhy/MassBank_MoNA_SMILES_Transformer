"""
Model file for MassBank CLIP-style Zero-Shot Retrieval (ZSR).
(MODIFIED for "Path A": Transformer-on-Peaks)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType


class MSEncoder(nn.Module):
    """
    (NEW "Path A" Encoder)
    Transformer Encoder for variable-length (m/z, int) peak sequences.
    """
    def __init__(self, encoder_config, embedding_dim):
        super().__init__()
        
        d_model = encoder_config['d_model']
        
        # --- 1. Peak Embedding ---
        # (m/z, int) 2D 입력을 d_model (768) 차원으로 임베딩
        self.peak_embed = nn.Linear(2, d_model)
        
        # --- 2. Positional Embedding ---
        # [CLS] 토큰 1개 + 최대 피크(MAX_PEAK_SEQ_LEN)
        seq_len = config.MAX_PEAK_SEQ_LEN
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # --- 3. Transformer Encoder ---
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_config['nhead'],
            dim_feedforward=d_model * 4,
            dropout=encoder_config['dropout'],
            batch_first=True # (Batch, SeqLen, Channels)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=encoder_config['n_layers']
        )
        
        # --- 4. Projection Head ---
        # (d_model -> embedding_dim)
        # config에서 d_model과 embedding_dim을 768로 통일했음
        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(encoder_config['dropout']),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x, mask):
        # x: [N, 700, 2] (정규화된 peak_sequence)
        # mask: [N, 700] (bool 마스크, True=실제 피크)
        
        # 1. Peak Embedding
        # (N, 700, 2) -> (N, 700, 768)
        x = self.peak_embed(x)
        
        # 2. [CLS] 토큰 추가
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [N, 1, 768]
        x = torch.cat((cls_tokens, x), dim=1) # [N, 701, 768]
        
        # 3. 위치 임베딩 추가
        x = x + self.pos_embedding
        
        # 4. 트랜스포머용 최종 마스크 생성
        # [CLS] 토큰은 항상 유효(True)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device) # [N, 1]
        full_mask = torch.cat((cls_mask, mask), dim=1) # [N, 701]
        
        # [중요] TransformerEncoder는 "True"를 패딩(무시)으로 간주함.
        # 우리의 마스크(True=유효)와 반대이므로, 마스크를 반전(~).
        transformer_mask = ~full_mask # [N, 701]
        
        # 5. 트랜스포머 인코더 실행
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        # 6. [CLS] 토큰의 출력만 사용
        cls_output = x[:, 0, :] # [N, 768]
        
        # 7. Projection
        x = self.projection(cls_output)
        
        # 8. L2 정규화
        x = F.normalize(x, p=2, dim=1)
        return x

class TextEncoder(nn.Module):
    """
    (수정 없음 - 원본과 동일)
    """
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

class CLIPModel(nn.Module):
    """
    (MODIFIED)
    MS Encoder를 새 ViT 모델로 교체
    """
    def __init__(self):
        super().__init__()
        self.ms_encoder = MSEncoder(config.MS_ENCODER, config.EMBEDDING_DIM)
        self.text_encoder = TextEncoder(
            config.TEXT_ENCODER['model_name'],
            config.EMBEDDING_DIM
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.TEMPERATURE))

    def forward(self, peak_sequence, peak_mask, input_ids, attention_mask):
        # --- [수정] peak_sequence, peak_mask를 ms_encoder로 전달 ---
        spec_embeds = self.ms_encoder(peak_sequence, peak_mask)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        # --- (이하 Loss 계산은 동일) ---
        batch_size = spec_embeds.shape[0]
        logits_per_spec = (spec_embeds @ text_embeds.T) * self.logit_scale.exp()
        logits_per_text = logits_per_spec.T
        labels = torch.arange(batch_size, device=config.DEVICE, dtype=torch.long)
        
        loss_spec = F.cross_entropy(logits_per_spec, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_spec + loss_text) / 2
        
        return loss