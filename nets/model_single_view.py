import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from einops import rearrange
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

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + attn_output

        x_ffn_in = x

        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        x = x + x_ffn_in

        return x


class InterleavedSpatioTemporalBlock(nn.Module):
    """
    Inside each block: temporal attention → spatial attention
    [CLS] tokens participate throughout
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


class ResnetTransformerDualTokensTemporalSpatialDecouplesize(nn.Module):
    def __init__(self, num_classes=18, d_model_cnn=512,
                 nhead=8, num_layers=6, dropout=0.1, spatial_resolution=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # CNN components
        self.spatial_size = spatial_resolution
        self.num_patch_tokens = self.spatial_size * self.spatial_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model_cnn, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))

        # Transformer components
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, d_model_cnn))
        self.pos_encoder_patch = nn.Parameter(torch.randn(1, self.num_patch_tokens, d_model_cnn))
        self.pos_encoder_temporal = PositionalEncoding(d_model_cnn, dropout, max_len=32)

        # Stack interleaved spatio-temporal blocks
        self.transformer_blocks = nn.ModuleList(
            [InterleavedSpatioTemporalBlock(d_model_cnn, nhead, d_model_cnn * 4, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.cls_head_norm = nn.LayerNorm(d_model_cnn)
        self.cls_head_projection = nn.Linear(d_model_cnn, 1)

    def forward(self, x, labels_18cls, doppler_mask=None):
        b, t, c, h_in, w_in = x.shape

        # CNN feature extraction
        x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
        feature_maps = self.cnn_backbone(x_reshaped)
        projected_maps = self.projection(feature_maps)
        pooled_maps = self.pool(projected_maps)

        # Prepare raw strings for loss computation
        raw_strings_for_loss = rearrange(pooled_maps.detach(), '(b t) c h w -> b (h w) t c', t=t)

        # Prepare Transformer input
        patch_tokens = rearrange(pooled_maps, '(b t) c h w -> b t (h w) c', t=t)
        cls_tokens = self.class_tokens.unsqueeze(1).expand(b, t, -1, -1)

        # Add spatial positional encoding
        patch_tokens = patch_tokens + self.pos_encoder_patch.unsqueeze(1)

        # Concatenate CLS and patch tokens
        initial_x = torch.cat([cls_tokens, patch_tokens], dim=2)

        # Add temporal positional encoding
        s = initial_x.shape[2]
        x_temp_for_pe = rearrange(initial_x, 'b t s c -> (b s) t c')
        x_temp_for_pe = self.pos_encoder_temporal(x_temp_for_pe)
        current_x = rearrange(x_temp_for_pe, '(b s) t c -> b t s c', s=s)

        # Interleaved Transformer encoding
        all_layer_cls_tokens = []
        all_layer_spatial_scores = []
        total_decouple_loss = torch.tensor(0.0, device=x.device)

        for block in self.transformer_blocks:
            s = current_x.shape[2]
            x_in_block = current_x.clone()

            # Temporal attention
            x_temp = rearrange(current_x, 'b t s c -> (b s) t c')
            x_temp = block.temporal_transformer(x_temp)
            current_x = rearrange(x_temp, '(b s) t c -> b t s c', s=s)

            # Spatial attention
            x_spat = rearrange(current_x, 'b t s c -> (b t) s c')

            if doppler_mask is not None:
                cls_mask = torch.zeros(b, t, self.num_classes, dtype=torch.bool, device=x.device)
                patch_mask = doppler_mask.unsqueeze(1).expand(-1, t, -1)
                spatial_padding_mask = torch.cat([cls_mask, patch_mask], dim=2)
                spatial_padding_mask = rearrange(spatial_padding_mask, 'b t s -> (b t) s')
            else:
                spatial_padding_mask = None

            x_spat = block.spatial_transformer(x_spat, src_key_padding_mask=spatial_padding_mask)
            current_x = rearrange(x_spat, '(b t) s c -> b t s c', t=t)

            # Extract CLS and patch tokens
            current_cls_temporal = current_x[:, :, :self.num_classes, :]
            current_patch_temporal = current_x[:, :, self.num_classes:, :]

            # Temporal aggregation
            current_cls_spatial = current_cls_temporal.mean(dim=1)
            current_patch_spatial = current_patch_temporal.mean(dim=1)

            # Compute decoupling loss
            norm_tokens = F.normalize(current_cls_spatial, p=2, dim=-1, eps=1e-6)
            similarity_matrix = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
            ground_truth = torch.arange(self.num_classes, device=x.device)

            loss_decouple_layer_i = F.cross_entropy(
                similarity_matrix.view(-1, self.num_classes),
                ground_truth.repeat(b)
            )
            total_decouple_loss += loss_decouple_layer_i

            # Compute attention scores in FP32 to avoid overflow
            with torch.autocast(device_type='cuda', enabled=False):
                scale_factor = current_cls_spatial.size(-1) ** 0.5
                cls_fp32 = current_cls_spatial.float()
                patch_fp32 = current_patch_spatial.float()
                cls_scaled = cls_fp32 / scale_factor
                scores = torch.einsum('bic,bpc->bip', cls_scaled, patch_fp32)

            all_layer_spatial_scores.append(scores)
            all_layer_cls_tokens.append(current_cls_spatial)

        # Final classification
        final_output_cls_tokens = all_layer_cls_tokens[-1]
        normed_cls_tokens = self.cls_head_norm(final_output_cls_tokens)
        projected_scores = self.cls_head_projection(normed_cls_tokens)
        predicted_scores = projected_scores.squeeze(-1)

        return {
            "predicted_scores": predicted_scores,
            "final_cls_tokens": final_output_cls_tokens,
            "all_layer_cls_tokens": all_layer_cls_tokens,
            "raw_strings_for_loss": raw_strings_for_loss,
            "all_layer_attention_scores": all_layer_spatial_scores,
            "total_decouple_loss": total_decouple_loss,
        }


class ResnetTransformerDualTokensTemporalSpatialDecouplesizeGradCam(nn.Module):
    def __init__(self, num_classes=18, d_model_cnn=512,
                 nhead=8, num_layers=6, dropout=0.1, spatial_resolution=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # CNN components
        self.spatial_size = spatial_resolution
        self.num_patch_tokens = self.spatial_size * self.spatial_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model_cnn, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))

        # Transformer components
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, d_model_cnn))
        self.pos_encoder_patch = nn.Parameter(torch.randn(1, self.num_patch_tokens, d_model_cnn))
        self.pos_encoder_temporal = PositionalEncoding(d_model_cnn, dropout, max_len=16)

        # Stack interleaved spatio-temporal blocks
        self.transformer_blocks = nn.ModuleList(
            [InterleavedSpatioTemporalBlock(d_model_cnn, nhead, d_model_cnn * 4, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.cls_head_norm = nn.LayerNorm(d_model_cnn)
        self.cls_head_projection = nn.Linear(d_model_cnn, 1)

        # Register target attention layer for Grad-CAM
        self.target_spatial_transformer = self.transformer_blocks[-1].spatial_transformer.attn

    def forward(self, x, labels_18cls, doppler_mask=None):
        b, t, c, h_in, w_in = x.shape

        # CNN feature extraction
        x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
        feature_maps = self.cnn_backbone(x_reshaped)
        projected_maps = self.projection(feature_maps)
        pooled_maps = self.pool(projected_maps)

        # Prepare raw strings for loss computation
        raw_strings_for_loss = rearrange(pooled_maps.detach(), '(b t) c h w -> b (h w) t c', t=t)

        # Prepare Transformer input
        patch_tokens = rearrange(pooled_maps, '(b t) c h w -> b t (h w) c', t=t)
        cls_tokens = self.class_tokens.unsqueeze(1).expand(b, t, -1, -1)

        # Add spatial positional encoding
        patch_tokens = patch_tokens + self.pos_encoder_patch.unsqueeze(1)

        # Concatenate CLS and patch tokens
        initial_x = torch.cat([cls_tokens, patch_tokens], dim=2)

        # Add temporal positional encoding
        s = initial_x.shape[2]
        x_temp_for_pe = rearrange(initial_x, 'b t s c -> (b s) t c')
        x_temp_for_pe = self.pos_encoder_temporal(x_temp_for_pe)
        current_x = rearrange(x_temp_for_pe, '(b s) t c -> b t s c', s=s)

        # Interleaved Transformer encoding
        all_layer_cls_tokens = []
        all_layer_spatial_scores = []
        total_decouple_loss = torch.tensor(0.0, device=x.device)

        for block in self.transformer_blocks:
            s = current_x.shape[2]
            x_in_block = current_x.clone()

            # Temporal attention
            x_temp = rearrange(current_x, 'b t s c -> (b s) t c')
            x_temp = block.temporal_transformer(x_temp)
            current_x = rearrange(x_temp, '(b s) t c -> b t s c', s=s)

            # Spatial attention
            x_spat = rearrange(current_x, 'b t s c -> (b t) s c')

            if doppler_mask is not None:
                cls_mask = torch.zeros(b, t, self.num_classes, dtype=torch.bool, device=x.device)
                patch_mask = doppler_mask.unsqueeze(1).expand(-1, t, -1)
                spatial_padding_mask = torch.cat([cls_mask, patch_mask], dim=2)
                spatial_padding_mask = rearrange(spatial_padding_mask, 'b t s -> (b t) s')
            else:
                spatial_padding_mask = None

            x_spat = block.spatial_transformer(x_spat, src_key_padding_mask=spatial_padding_mask)
            current_x = rearrange(x_spat, '(b t) s c -> b t s c', t=t)

            # Extract CLS and patch tokens
            current_cls_temporal = current_x[:, :, :self.num_classes, :]
            current_patch_temporal = current_x[:, :, self.num_classes:, :]

            # Temporal aggregation
            current_cls_spatial = current_cls_temporal.mean(dim=1)
            current_patch_spatial = current_patch_temporal.mean(dim=1)

            # Compute decoupling loss
            norm_tokens = F.normalize(current_cls_spatial, p=2, dim=-1)
            similarity_matrix = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
            ground_truth = torch.arange(self.num_classes, device=x.device)

            loss_decouple_layer_i = F.cross_entropy(
                similarity_matrix.view(-1, self.num_classes),
                ground_truth.repeat(b)
            )
            total_decouple_loss += loss_decouple_layer_i

            # Compute attention scores
            scores = torch.einsum('bic,bpc->bip', current_cls_spatial, current_patch_spatial)
            all_layer_spatial_scores.append(scores)
            all_layer_cls_tokens.append(current_cls_spatial)

        # Final classification
        final_output_cls_tokens = all_layer_cls_tokens[-1]
        normed_cls_tokens = self.cls_head_norm(final_output_cls_tokens)
        projected_scores = self.cls_head_projection(normed_cls_tokens)
        predicted_scores = projected_scores.squeeze(-1)

        return {
            "predicted_scores": predicted_scores,
            "final_cls_tokens": final_output_cls_tokens,
            "all_layer_cls_tokens": all_layer_cls_tokens,
            "raw_strings_for_loss": raw_strings_for_loss,
            "all_layer_attention_scores": all_layer_spatial_scores,
            "total_decouple_loss": total_decouple_loss,
        }