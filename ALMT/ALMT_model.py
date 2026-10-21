from __future__ import annotations
import copy
import torch
from torch import nn
from mosei_dataset import FEATURE_DIMS
from .ALMT_data import MODALITIES

class ModalityEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        token_count: int,
        max_input_length: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.token_count = token_count
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.latent_tokens = nn.Parameter(torch.empty(1, token_count, hidden_dim))
        self.position_embeddings = nn.Parameter(torch.empty(1, max_input_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.shape[1] > self.position_embeddings.shape[1]:
            raise ValueError(
                f"Input sequence length {features.shape[1]} exceeds configured "
                f"maximum {self.position_embeddings.shape[1]}."
            )
        projected = self.input_projection(features)
        projected = projected + self.position_embeddings[:, : projected.shape[1]]
        latents = self.latent_tokens.expand(features.shape[0], -1, -1)
        combined = torch.cat((latents, projected), dim=1)
        latent_padding = torch.zeros(
            (features.shape[0], self.token_count), dtype=torch.bool, device=features.device
        )
        padding_mask = torch.cat((latent_padding, ~valid_mask.bool()), dim=1)
        encoded = self.transformer(combined, src_key_padding_mask=padding_mask)
        return encoded[:, : self.token_count]

class AdaptiveHyperModalityLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.language_to_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.language_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        language_scale: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        audio_padding_mask: torch.Tensor,
        visual_padding_mask: torch.Tensor,
        hyper: torch.Tensor,
    ) -> torch.Tensor:
        audio_context, _ = self.language_to_audio(
            query=language_scale,
            key=audio,
            value=audio,
            key_padding_mask=audio_padding_mask,
            need_weights=False,
        )
        visual_context, _ = self.language_to_visual(
            query=language_scale,
            key=visual,
            value=visual,
            key_padding_mask=visual_padding_mask,
            need_weights=False,
        )
        return self.norm(hyper + self.dropout(audio_context) + self.dropout(visual_context))

class CrossModalFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_feed_forward = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        query = self.norm_attention(query + self.dropout(attended))
        return self.norm_feed_forward(query + self.dropout(self.feed_forward(query)))

class MultimodalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Parameter(torch.zeros(5, 5))
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = nodes.shape[-1] ** -0.5
        edge_logits = torch.matmul(self.query(nodes), self.key(nodes).transpose(1, 2)) * scale
        edge_weights = torch.softmax(edge_logits + self.edge_bias, dim=-1)
        aggregated = torch.matmul(edge_weights, self.message(nodes))
        updated = self.message_norm(
            self.self_projection(nodes) + self.dropout(aggregated)
        )
        updated = self.output_norm(updated + self.dropout(self.feed_forward(updated)))
        return updated, edge_weights

class MultimodalGraphRegressor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, score_limit: float = 3.0, output_dim: int = 1):
        super().__init__()
        self.graph_layers = nn.ModuleList(
            [MultimodalGraphLayer(hidden_dim, dropout) for _ in range(2)]
        )
        self.pool_query = nn.Parameter(torch.empty(hidden_dim))
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.score_limit = score_limit
        self.is_classifier = output_dim > 1
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        for layer in self.graph_layers:
            nodes, _ = layer(nodes)
        pooling_scores = torch.matmul(nodes, self.pool_query) / (nodes.shape[-1] ** 0.5)
        node_weights = torch.softmax(pooling_scores, dim=1).unsqueeze(-1)
        graph_representation = (nodes * node_weights).sum(dim=1)
        raw_output = self.regression_head(graph_representation)
        if self.is_classifier:
            return raw_output
        raw_score = raw_output.squeeze(-1)
        return self.score_limit * torch.tanh(raw_score / self.score_limit)

class MSALMT(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int] | None = None,
        hidden_dim: int = 128,
        token_count: int = 8,
        max_input_length: int = 256,
        num_heads: int = 8,
        ahl_depth: int = 3,
        fusion_depth: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1,
        use_modality_embedding: bool = True,
        use_ahl: bool = True,
        use_cross_modal_fusion: bool = True,
        use_graph_fusion: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if token_count < 1 or ahl_depth != 3 or fusion_depth < 1:
            raise ValueError("token_count/fusion_depth must be positive; ALMT uses three AHL scales.")
        self.modality_embedding = nn.ModuleDict(
            {
                modality: ModalityEmbedding(
                    (feature_dims or FEATURE_DIMS)[modality],
                    hidden_dim,
                    token_count,
                    max_input_length,
                    num_heads,
                    dropout,
                )
                for modality in MODALITIES
            }
        )
        language_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.language_scales = nn.ModuleList(
            [copy.deepcopy(language_layer) for _ in range(2)]
        )
        self.language_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.initial_hyper_tokens = nn.Parameter(torch.empty(1, token_count, hidden_dim))
        self.ahl_layers = nn.ModuleList(
            [AdaptiveHyperModalityLayer(hidden_dim, num_heads, dropout) for _ in range(ahl_depth)]
        )
        self.fusion_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.fusion_layers = nn.ModuleList(
            [CrossModalFusionLayer(hidden_dim, num_heads, dropout) for _ in range(fusion_depth)]
        )
        self.regressor = MultimodalGraphRegressor(
            hidden_dim, dropout, output_dim=output_dim
        )
        self.raw_projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear((feature_dims or FEATURE_DIMS)[modality], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for modality in MODALITIES
            }
        )
        self.non_graph_regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        self.use_modality_embedding = use_modality_embedding
        self.use_ahl = use_ahl
        self.use_cross_modal_fusion = use_cross_modal_fusion
        self.use_graph_fusion = use_graph_fusion
        nn.init.normal_(self.initial_hyper_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.fusion_token, mean=0.0, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_modality_embedding:
            embedded = {
                modality: self.modality_embedding[modality](
                    batch[f"{modality}_features"], batch[f"{modality}_mask"]
                )
                for modality in MODALITIES
            }
        else:
            embedded = {}
            for modality in MODALITIES:
                features = batch[f"{modality}_features"]
                valid_mask = batch[f"{modality}_mask"].to(features.dtype).unsqueeze(-1)
                projected = self.raw_projections[modality](features)
                pooled = (projected * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
                embedded[modality] = pooled.unsqueeze(1).expand(
                    -1, self.initial_hyper_tokens.shape[1], -1
                )
        language_scales = [embedded["text"]]
        for layer, norm in zip(self.language_scales, self.language_norms):
            language_scales.append(norm(layer(language_scales[-1])))

        audio = embedded["audio"]
        visual = embedded["image"]
        hyper = self.initial_hyper_tokens.expand(audio.shape[0], -1, -1)
        all_valid = torch.zeros(
            (audio.shape[0], audio.shape[1]), dtype=torch.bool, device=audio.device
        )
        if self.use_ahl:
            for language, ahl_layer in zip(language_scales, self.ahl_layers):
                hyper = ahl_layer(language, audio, visual, ~all_valid, ~all_valid, hyper)
        else:
            hyper = 0.5 * (audio + visual)

        if self.use_cross_modal_fusion:
            fusion_query = torch.cat(
                (self.fusion_token.expand(audio.shape[0], -1, -1), language_scales[-1]), dim=1
            )
            fusion_key_value = torch.cat(
                (self.fusion_token.expand(audio.shape[0], -1, -1), hyper), dim=1
            )
            for layer in self.fusion_layers:
                fusion_query = layer(fusion_query, fusion_key_value)
            fusion_summary = fusion_query[:, 0]
        else:
            fusion_summary = language_scales[-1].mean(dim=1)
        graph_nodes = torch.stack(
            (
                language_scales[-1].mean(dim=1),
                audio.mean(dim=1),
                visual.mean(dim=1),
                hyper.mean(dim=1),
                fusion_summary,
            ),
            dim=1,
        )
        if self.use_graph_fusion:
            return self.regressor(graph_nodes)
        raw_output = self.non_graph_regressor(fusion_summary)
        if self.output_dim > 1:
            return raw_output
        return raw_output.squeeze(-1)
