import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from einops import rearrange, repeat
from copy import deepcopy
from entmax import EntmaxBisect


# Standard positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


class BasicTransformerBlock(nn.Module):
    """Standard Transformer encoder layer"""
    def __init__(self, dim, n_heads, dim_feedforward, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, src_key_padding_mask=None):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + attn_output
        x = x + self.ff(self.norm2(x))
        return x


# Interleaved spatio-temporal Transformer block
class InterleavedSpatioTemporalBlock(nn.Module):
    """
    Inside each block: temporal attention → spatial attention
    [CLS] tokens participate throughout
    Note: residual connection needs refinement
    """
    def __init__(self, dim, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.temporal_transformer = BasicTransformerBlock(dim, n_heads, dim_feedforward, dropout)
        self.spatial_transformer = BasicTransformerBlock(dim, n_heads, dim_feedforward, dropout)

    def forward(self, x, t, h, w):
        b, c, t, h, w = x.shape
        x_in = x

        # Temporal attention
        x_temporal = rearrange(x, 'b c t h w -> (b h w) t c')
        x_temporal = self.temporal_transformer(x_temporal)
        x_temporal = rearrange(x_temporal, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
        x = x_in + x_temporal

        # Spatial attention
        x_in_spatial = x
        x_spatial = rearrange(x, 'b c t h w -> (b t) c (h w)')
        x_spatial = x_spatial.permute(0, 2, 1)
        x_spatial = self.spatial_transformer(x_spatial)
        x_spatial = x_spatial.permute(0, 2, 1)
        x_spatial = rearrange(x_spatial, '(b t) c (h w) -> b c t h w', t=t, h=h, w=w)
        x = x_in_spatial + x_spatial

        return x


class MultiViewDualTokensFusionSize(nn.Module):
    def __init__(self, view_num_classes=18, d_model=512,
                 view_nhead=8, view_encoder_layers=6,
                 case_num_classes=4,
                 fusion_layers=2, fusion_nhead=8,
                 max_views=5,
                 dropout=0.1,
                 spatial_size=4):
        super().__init__()

        # Parameters
        self.view_num_classes = view_num_classes
        self.case_num_classes = case_num_classes
        self.num_patch_tokens = spatial_size * spatial_size

        # Shared CNN backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))

        # View-level encoder components
        self.cls_tokens = nn.Parameter(torch.randn(1, self.view_num_classes, d_model))
        self.pos_encoder_patch = nn.Parameter(torch.randn(1, self.num_patch_tokens, d_model))
        self.pos_encoder_temporal = PositionalEncoding(d_model, dropout, max_len=32)

        # Spatio-temporal encoder blocks
        self.transformer_blocks = nn.ModuleList(
            [InterleavedSpatioTemporalBlock(d_model, view_nhead, d_model * 4, dropout) for _ in range(view_encoder_layers)]
        )

        # Cross-view fusion components
        self.case_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        max_fusion_sequence_len = 1 + max_views * self.view_num_classes
        self.view_pos_embedding = nn.Parameter(torch.randn(1, max_fusion_sequence_len, d_model))

        # Fusion Transformer
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model, fusion_nhead, d_model * 4, dropout, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=fusion_layers)

        # Final classification head
        self.case_head_norm = nn.LayerNorm(d_model)
        self.case_head_projection = nn.Linear(d_model, self.case_num_classes)

    def forward(self, x_case, num_views_per_sample, masks):
        b, v, t, c, h, w = x_case.shape

        # --------------------------
        # Single-view encoding
        # --------------------------
        x_reshaped = rearrange(x_case, 'b v t c h w -> (b v t) c h w')
        pooled_maps = self.pool(self.projection(self.cnn_backbone(x_reshaped)))
        patch_tokens = rearrange(pooled_maps, '(b v t) d h w -> (b v) t (h w) d', b=b, v=v, t=t)

        patch_tokens = patch_tokens + self.pos_encoder_patch
        view_cls_tokens_expanded = self.cls_tokens.expand(b*v, -1, -1).unsqueeze(1).expand(-1, t, -1, -1)
        current_sequence = torch.cat([view_cls_tokens_expanded, patch_tokens], dim=2)

        # Spatial attention mask
        spatial_attn_mask = None
        if masks is not None:
            masks_reshaped_bv = rearrange(masks, 'b v s -> (b v) s')
            masks_reshaped_bvt = repeat(masks_reshaped_bv, 'bv s -> (bv t) s', t=t)
            cls_mask = torch.zeros(b*v*t, self.view_num_classes, dtype=torch.bool, device=x_case.device)
            spatial_attn_mask = torch.cat([cls_mask, masks_reshaped_bvt], dim=1)

        # Spatio-temporal Transformer blocks
        s_inner = current_sequence.shape[2]
        for block in self.transformer_blocks:
            # Temporal
            x_for_temporal = rearrange(current_sequence, 'bv t s d -> (bv s) t d')
            x_for_temporal = self.pos_encoder_temporal(x_for_temporal)
            temporal_out = block.temporal_transformer(x_for_temporal)
            current_sequence = rearrange(temporal_out, '(bv s) t d -> bv t s d', s=s_inner)

            # Spatial
            x_for_spatial = rearrange(current_sequence, 'bv t s d -> (bv t) s d')
            spatial_out = block.spatial_transformer(x_for_spatial, src_key_padding_mask=spatial_attn_mask)
            current_sequence = rearrange(spatial_out, '(bv t) s d -> bv t s d', t=t)

        # --------------------------
        # View feature extraction
        # --------------------------
        final_view_cls = current_sequence[:, :, :self.view_num_classes, :]
        view_features_set = final_view_cls.mean(dim=1)

        # --------------------------
        # Cross-view fusion
        # --------------------------
        view_features_sequence = rearrange(view_features_set, '(b v) s d -> b (v s) d', b=b)
        case_cls = self.case_cls_token.expand(b, -1, -1)
        fusion_sequence = torch.cat([case_cls, view_features_sequence], dim=1)

        # Position embedding
        num_fusion_tokens = fusion_sequence.shape[1]
        fusion_sequence = fusion_sequence + self.view_pos_embedding[:, :num_fusion_tokens, :]

        # Fusion mask
        fusion_mask = None
        if num_views_per_sample is not None:
            view_token_mask = repeat(
                torch.arange(v, device=x_case.device) >= num_views_per_sample.unsqueeze(1),
                'b v -> b (v s)', s=self.view_num_classes
            )
            cls_padding_mask = torch.zeros(b, 1, dtype=torch.bool, device=x_case.device)
            fusion_mask = torch.cat([cls_padding_mask, view_token_mask], dim=1)

        # Fusion Transformer
        fused_output = self.fusion_transformer(fusion_sequence, src_key_padding_mask=fusion_mask)

        # --------------------------
        # Final case-level prediction
        # --------------------------
        final_case_token = fused_output[:, 0, :]
        logits = self.case_head_projection(self.case_head_norm(final_case_token))

        return logits