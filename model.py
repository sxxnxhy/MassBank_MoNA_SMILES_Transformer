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
import math

class GaussianFourierProjection(nn.Module):
    """
    Projects scalar m/z values into high-dimensional space using 
    random Gaussian frequencies (Tancik et al., NeurIPS 2020).
    This enables the MLP to learn high-frequency functions (fine mass differences).
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Random B matrix (fixed, not learnable)
        # scale controls the frequency spectrum. Higher scale = higher freq sensitivity.
        self.B = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [Batch, Seq, 1]
        # x_proj: [Batch, Seq, embed_dim/2]
        x_proj = (2 * np.pi * x) @ self.B 
        # cat: [Batch, Seq, embed_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MSEncoder(nn.Module):
    """
    Physics-Informed Transformer.
    Input: (m/z, intensity) pairs.
    Architecture:
      1. m/z -> Fourier Projection -> MLP
      2. intensity -> MLP
      3. Concat([mz_emb, int_emb]) -> Fusion -> Transformer
    """
    def __init__(self, encoder_config, embedding_dim):
        super().__init__()
        
        d_model = encoder_config['d_model']
        fourier_dim = encoder_config['fourier_dim']
        
        # --- 1. m/z Pathway (Where?) ---
        self.mz_fourier = GaussianFourierProjection(fourier_dim, scale=30.0)
        self.mz_mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # --- 2. Intensity Pathway (How much?) ---
        self.int_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # --- 3. Fusion Layer (Concatenation Strategy) ---
        # Takes [mz_emb; int_emb] -> Fuses to [d_model]
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # --- 4. Transformer ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
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
        
        # --- 5. Projection ---
        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(encoder_config['dropout']),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x, mask):
        # x: [B, S, 2] -> (log_mz, sqrt_int)
        mz_val = x[:, :, 0:1]
        int_val = x[:, :, 1:2]
        
        # 1. Encode m/z (Fourier -> MLP)
        mz_emb = self.mz_fourier(mz_val) # [B, S, fourier_dim]
        mz_emb = self.mz_mlp(mz_emb)     # [B, S, d_model]
        
        # 2. Encode intensity (MLP)
        int_emb = self.int_mlp(int_val)  # [B, S, d_model]
        
        # 3. Concatenate & Fuse (FIX 4)
        # Explicitly separate location and magnitude information
        cat_feat = torch.cat([mz_emb, int_emb], dim=-1) # [B, S, 2*d]
        x_emb = self.fusion(cat_feat) # [B, S, d]
        
        # 4. Append CLS
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # 5. Masking
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)
        transformer_mask = ~full_mask
        
        # 6. Transformer
        out = self.transformer_encoder(x_emb, src_key_padding_mask=transformer_mask)
        
        # 7. Output
        cls_output = out[:, 0, :]
        projected = self.projection(cls_output)
        return F.normalize(projected, p=2, dim=1)

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