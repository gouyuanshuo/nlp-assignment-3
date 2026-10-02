"""The three classifiers share cached inputs, masks, and two-logit outputs."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .data import MODALITIES


@dataclass(frozen=True)
class ModelConfig:
    name: str = "transformer"
    d_model: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    modality_embeddings: bool = True
    positions: bool = True
    pooling: str = "cls"

    def __post_init__(self):
        if self.name not in {"text", "concat", "transformer"}:
            raise ValueError(f"Unknown model: {self.name}")
        if min(self.d_model, self.hidden_dim, self.n_layers, self.n_heads, self.ff_dim) < 1:
            raise ValueError("Model dimensions/layers/heads must be positive")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if not 0 <= self.dropout < 1 or self.pooling not in {"cls", "mean"}:
            raise ValueError("Invalid dropout or pooling")


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = features.masked_fill(~mask[..., None], 0)
    return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def _head(width: int, config: ModelConfig):
    return nn.Sequential(
        nn.Linear(width, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, 2),
    )


class TextClassifier(nn.Module):
    def __init__(self, config, feature_dims):
        super().__init__()
        self.head = _head(feature_dims["text"], config)

    def forward(self, features, masks):
        return self.head(masked_mean(features["text"], masks["text"]))


class ConcatClassifier(nn.Module):
    def __init__(self, config, feature_dims):
        super().__init__()
        self.projections = nn.ModuleDict(
            {m: nn.Linear(feature_dims[m], config.d_model) for m in MODALITIES}
        )
        self.head = _head(3 * config.d_model, config)

    def forward(self, features, masks):
        pooled = []
        for m in MODALITIES:
            safe = features[m].masked_fill(~masks[m][..., None], 0)
            pooled.append(masked_mean(self.projections[m](safe), masks[m]))
        return self.head(torch.cat(pooled, dim=-1))


def sinusoidal_positions(length: int, width: int, device, dtype):
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000) / width))
    angles = positions * frequencies
    result = torch.zeros(length, width, device=device)
    result[:, 0::2], result[:, 1::2] = angles.sin(), angles.cos()
    return result.to(dtype=dtype)


class TransformerClassifier(nn.Module):
    def __init__(self, config, feature_dims):
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict(
            {m: nn.Linear(feature_dims[m], config.d_model) for m in MODALITIES}
        )
        self.cls = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.modality = nn.Embedding(3, config.d_model) if config.modality_embeddings else None
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.ff_dim,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            config.n_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        # PyTorch clones a layer; initialize each clone independently.
        for encoder_layer in self.encoder.layers:
            for module in encoder_layer.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    module.reset_parameters()
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            nn.init.zeros_(encoder_layer.self_attn.in_proj_bias)
        self.head = _head(config.d_model, config)

    def forward(self, features, masks):
        first = features["text"]
        tokens = [self.cls.expand(first.shape[0], -1, -1)]
        valid = [torch.ones(first.shape[0], 1, dtype=torch.bool, device=first.device)]
        for index, m in enumerate(MODALITIES):
            safe = features[m].masked_fill(~masks[m][..., None], 0)
            projected = self.projections[m](safe)
            if self.modality is not None:
                projected = projected + self.modality.weight[index]
            if self.config.positions:
                projected = projected + sinusoidal_positions(
                    projected.shape[1],
                    projected.shape[2],
                    projected.device,
                    projected.dtype,
                )
            tokens.append(projected.masked_fill(~masks[m][..., None], 0))
            valid.append(masks[m])
        tokens, valid = torch.cat(tokens, dim=1), torch.cat(valid, dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        if self.config.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            # Exclude CLS from mean; the all-missing case has a defined zero vector.
            pooled = masked_mean(encoded[:, 1:], valid[:, 1:])
        return self.head(pooled)


def build_model(config: ModelConfig, feature_dims: dict[str, int]) -> nn.Module:
    if set(feature_dims) != set(MODALITIES) or min(feature_dims.values()) < 1:
        raise ValueError("Positive feature widths for text, audio and vision are required")
    return {
        "text": TextClassifier,
        "concat": ConcatClassifier,
        "transformer": TransformerClassifier,
    }[config.name](config, feature_dims)
