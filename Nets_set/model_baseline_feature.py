import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from einops import rearrange, repeat
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


class ResNetTransformerDualClsEntmaxWithDopplerMaskTemporal(nn.Module):
    """
    Video classification model with [CLS] token guidance and Entmax sparse pooling.
    Best visualization performance so far.
    Used to inspect spatial attention in internal layers.
    """
    def __init__(self, num_classes, d_model=512,
                 temporal_nhead=8, temporal_layers=6,
                 spatial_nhead=8, spatial_layers=12,
                 dim_feedforward=2048, dropout=0.1,
                 temporal_entmax_alpha=1.5, spatial_entmax_alpha=2.0):
        super().__init__()

        # CNN backbone (pretrained ResNet50, remove final avg pool and fc)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Projection layer: 2048 -> d_model
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)

        # --- Temporal Transformer ---
        self.cls_token_temporal = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder_temporal = PositionalEncoding(d_model, dropout=dropout, max_len=50)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model, temporal_nhead, dim_feedforward, dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_layers)
        self.temporal_sparse_activation = EntmaxBisect(alpha=temporal_entmax_alpha, dim=-1)

        # Classification head
        self.temporal_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        self.temporal_weights = None

    def forward(self, x, doppler_mask_7x7=None, return_attention=False):
        b, t, c, h, w = x.shape

        # CNN feature extraction
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        feature_maps = self.cnn_backbone(x)
        feature_maps = self.projection(feature_maps)
        frame_tokens = rearrange(feature_maps, '(b t) d h w -> b t (h w) d', t=t)

        # --- Temporal filtering based on blood flow area ---
        temporal_padding_mask = None
        if doppler_mask_7x7 is not None:
            # Compute blood flow area per frame
            blood_flow_areas = (~doppler_mask_7x7).float().sum(dim=(-1, -2))  # (B, T)

            # Dynamic threshold: 20% of max area
            max_areas, _ = blood_flow_areas.max(dim=1, keepdim=True)
            thresholds = max_areas * 0.2

            # Create mask: True = ignore
            temporal_padding_mask_for_frames = (blood_flow_areas < thresholds)
            cls_mask_temporal = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
            temporal_padding_mask = torch.cat((cls_mask_temporal, temporal_padding_mask_for_frames), dim=1)

        # --- Temporal attention ---
        cls_t = self.cls_token_temporal.expand(b, -1, -1)
        frame_level_features = frame_tokens.mean(dim=2)  # (B, T, D)
        temporal_input = torch.cat((cls_t, frame_level_features), dim=1)

        temporal_input = self.pos_encoder_temporal(temporal_input)
        temporal_output = self.temporal_transformer(temporal_input, src_key_padding_mask=temporal_padding_mask)

        # CLS-T guided temporal pooling
        cls_t_output = temporal_output[:, 0]
        frame_t_outputs = temporal_output[:, 1:]
        temporal_scores = torch.einsum('bd,btd->bt', cls_t_output, frame_t_outputs)
        temporal_weights = self.temporal_sparse_activation(temporal_scores)

        # Fuse temporal tokens
        fused_temporal_tokens = torch.einsum('btnd,bt->bnd', frame_tokens, temporal_weights)
        final_video_representation = fused_temporal_tokens.mean(dim=1)

        # Final classification
        logits_temporal = self.temporal_classifier(final_video_representation)

        if return_attention:
            return logits_temporal, temporal_weights.detach(), frame_t_outputs

        return logits_temporal, frame_t_outputs


class ResNetEncoder(nn.Module):
    """
    Simple ResNet encoder for video classification + contrastive learning.
    """
    def __init__(self, num_classes, d_model=512, projection_dim=128,
                 temporal_nhead=8, temporal_layers=6,
                 spatial_nhead=8, spatial_layers=12,
                 dim_feedforward=2048, dropout=0.1,
                 temporal_entmax_alpha=1.5, spatial_entmax_alpha=2.0):
        super().__init__()

        # CNN backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Projection layer
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )

    def forward(self, x, doppler_mask_7x7=None, return_attention=False):
        b, t, c, h, w = x.shape

        # CNN forward
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        feature_maps = self.cnn_backbone(x)
        feature_maps = self.projection(feature_maps)

        # Global average pooling -> frame-level vector
        frame_vectors = self.global_avg_pool(feature_maps)
        frame_vectors = torch.flatten(frame_vectors, 1)

        # Video-level feature
        video_sequence = rearrange(frame_vectors, '(b t) d -> b t d', b=b)
        base_features = video_sequence.mean(dim=1)

        # Classification branch
        logits = self.classifier(base_features)

        # Contrastive branch
        contrastive_features = self.projection_head(base_features)

        return logits, contrastive_features, base_features