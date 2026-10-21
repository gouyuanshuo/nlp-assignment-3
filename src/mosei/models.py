"""All classifiers share cached inputs, valid-token masks, and two-logit outputs."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from .data import MODALITIES


def _canonical_modalities(values) -> list[str]:
    raw = list(values)
    if not raw or any(item not in MODALITIES for item in raw) or len(raw) != len(set(raw)):
        raise ValueError("modalities must be a non-empty subset of text, audio and vision")
    return [item for item in MODALITIES if item in set(raw)]


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
    modalities: list[str] = field(default_factory=lambda: list(MODALITIES))
    almt_token_count: int = 8
    almt_max_input_length: int = 256
    almt_fusion_depth: int = 4
    almt_use_ahl: bool = True
    almt_use_cross_modal_fusion: bool = True
    almt_use_graph_fusion: bool = True
    graph_mfn_heads: int = 4
    graph_use_dynamic_graph: bool = True
    graph_use_view_memory: bool = True
    graph_use_fusion_memory: bool = True
    graph_use_temporal_attention: bool = True

    def __post_init__(self):
        if self.name not in {"text", "concat", "transformer", "almt", "graph_mfn"}:
            raise ValueError(f"Unknown model: {self.name}")
        if min(self.d_model, self.hidden_dim, self.n_layers, self.n_heads, self.ff_dim) < 1:
            raise ValueError("Model dimensions/layers/heads must be positive")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if not 0 <= self.dropout < 1 or self.pooling not in {"cls", "mean"}:
            raise ValueError("Invalid dropout or pooling")
        for name in (
            "almt_token_count",
            "almt_max_input_length",
            "almt_fusion_depth",
            "graph_mfn_heads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.name == "graph_mfn" and self.hidden_dim % self.graph_mfn_heads:
            raise ValueError("hidden_dim must be divisible by graph_mfn_heads")
        for name in (
            "almt_use_ahl",
            "almt_use_cross_modal_fusion",
            "almt_use_graph_fusion",
            "graph_use_dynamic_graph",
            "graph_use_view_memory",
            "graph_use_fusion_memory",
            "graph_use_temporal_attention",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        chosen = _canonical_modalities(self.modalities)
        if self.name == "text":
            if chosen not in (["text"], list(MODALITIES)):
                raise ValueError(
                    "Use concat or transformer to combine modalities; the text model reads text only"
                )
            chosen = ["text"]
        object.__setattr__(self, "modalities", chosen)


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = features.masked_fill(~mask[..., None], 0)
    return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def prepare_multimodal_inputs(features, masks, modalities, feature_dims):
    """Validate aligned inputs and erase masked values before any learned operation.

    The recurrent and ALMT ports retain three-stream architecture for ablations.
    A dropped stream is therefore represented by an entirely false mask, rather
    than allowing projection biases or NaN padding to introduce evidence.
    """
    first = features[MODALITIES[0]]
    if first.ndim != 3 or first.shape[1] < 1:
        raise ValueError("Expected nonempty aligned [batch, time, width] features")
    shape = first.shape[:2]
    safe, valid = {}, {}
    for modality in MODALITIES:
        values, mask = features[modality], masks[modality]
        if (
            values.ndim != 3
            or values.shape[:2] != shape
            or values.shape[2] != feature_dims[modality]
            or mask.shape != shape
        ):
            raise ValueError("Expected aligned feature sequences, masks and configured widths")
        valid[modality] = (
            mask.bool() if modality in modalities else torch.zeros_like(mask, dtype=torch.bool)
        )
        safe[modality] = values.masked_fill(~valid[modality][..., None], 0)
    return safe, valid


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
        self.modalities = list(config.modalities)
        self.projections = nn.ModuleDict(
            {m: nn.Linear(feature_dims[m], config.d_model) for m in self.modalities}
        )
        self.head = _head(len(self.modalities) * config.d_model, config)

    def forward(self, features, masks):
        pooled = []
        for m in self.modalities:
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
        self.modalities = list(config.modalities)
        self.projections = nn.ModuleDict(
            {m: nn.Linear(feature_dims[m], config.d_model) for m in self.modalities}
        )
        self.cls = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.modality = (
            nn.Embedding(len(self.modalities), config.d_model)
            if config.modality_embeddings
            else None
        )
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
        for index, m in enumerate(self.modalities):
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
    if config.name == "almt":
        from .almt import ALMTClassifier

        return ALMTClassifier(config, feature_dims)
    if config.name == "graph_mfn":
        from .graph_mfn import GraphMFNClassifier

        return GraphMFNClassifier(config, feature_dims)
    return {
        "text": TextClassifier,
        "concat": ConcatClassifier,
        "transformer": TransformerClassifier,
    }[config.name](config, feature_dims)
