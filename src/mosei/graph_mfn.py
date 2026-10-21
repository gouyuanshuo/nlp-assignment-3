"""Teammate Graph-MFN port, preserving text/audio/vision node and memory order.

The four switches reproduce the submitted component ablations without a second
trainer or model-input format. All predictions use two binary-class logits.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import MODALITIES
from .models import prepare_multimodal_inputs


def _masked_softmax(logits, valid, dim=-1):
    """No probability on padding, including an entirely missing row."""
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=dim).masked_fill(~valid, 0)
    denominator = weights.sum(dim=dim, keepdim=True)
    # Use one for an empty row: 1e-12 underflows to zero in float16.
    return weights / denominator.masked_fill(denominator == 0, 1)


class DynamicFusionGraph(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by a positive number of GAT heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, nodes: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, hidden_dim = nodes.shape
        projected = self.projection(nodes).view(
            batch_size, node_count, self.num_heads, self.head_dim
        )
        source_scores = (projected * self.attention_source).sum(dim=-1).permute(0, 2, 1)
        target_scores = (projected * self.attention_target).sum(dim=-1).permute(0, 2, 1)
        edge_logits = nn.functional.leaky_relu(
            source_scores.unsqueeze(-1) + target_scores.unsqueeze(-2),
            negative_slope=0.2,
        )
        edge_weights = self.dropout(_masked_softmax(edge_logits, valid[:, None, None, :]))
        values = projected.permute(0, 2, 1, 3)
        messages = torch.matmul(edge_weights, values)
        messages = messages.permute(0, 2, 1, 3).reshape(batch_size, node_count, hidden_dim)
        messages = messages * valid.unsqueeze(-1)
        graph_nodes = self.norm(nodes + self.dropout(self.output(messages)))
        graph_nodes = graph_nodes * valid.unsqueeze(-1)
        return graph_nodes, messages


class GraphMFNClassifier(nn.Module):
    def __init__(self, config, feature_dims: dict[str, int]):
        super().__init__()
        self.config = config
        self.feature_dims = dict(feature_dims)
        dims = feature_dims
        hidden_dim = config.hidden_dim
        dropout = config.dropout
        output_dim = 2
        gat_heads = config.graph_mfn_heads
        self.use_dynamic_graph = config.graph_use_dynamic_graph
        self.use_view_memory = config.graph_use_view_memory
        self.use_fusion_memory = config.graph_use_fusion_memory
        self.use_temporal_attention = config.graph_use_temporal_attention
        self.input_projection = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dims[modality], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for modality in MODALITIES
            }
        )
        self.unimodal_cells = nn.ModuleDict(
            {modality: nn.LSTMCell(hidden_dim, hidden_dim) for modality in MODALITIES}
        )
        self.dynamic_graph = DynamicFusionGraph(hidden_dim, dropout, gat_heads)
        self.view_memory_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim * 2, hidden_dim) for _ in MODALITIES]
        )
        self.fusion_memory = nn.GRUCell(hidden_dim * len(MODALITIES), hidden_dim)
        self.temporal_attention = nn.Linear(hidden_dim * (len(MODALITIES) + 1), 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * (len(MODALITIES) + 1)),
            nn.Linear(hidden_dim * (len(MODALITIES) + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, features, masks) -> torch.Tensor:
        features, masks = prepare_multimodal_inputs(
            features, masks, self.config.modalities, self.feature_dims
        )
        first_features = features[MODALITIES[0]]
        batch_size, time_steps = first_features.shape[:2]
        device = first_features.device
        dtype = first_features.dtype
        projected = {
            modality: self.input_projection[modality](features[modality]) for modality in MODALITIES
        }
        modality_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in MODALITIES
        ]
        modality_cells = [torch.zeros_like(modality_states[0]) for _ in MODALITIES]
        view_memories = [torch.zeros_like(modality_states[0]) for _ in MODALITIES]
        shared_memory = torch.zeros_like(modality_states[0])
        sequence_views: list[torch.Tensor] = []
        sequence_valid: list[torch.Tensor] = []

        for time_index in range(time_steps):
            current_valid = torch.stack(
                [masks[modality][:, time_index] for modality in MODALITIES], dim=1
            )
            for index, modality in enumerate(MODALITIES):
                candidate_state, candidate_cell = self.unimodal_cells[modality](
                    projected[modality][:, time_index],
                    (modality_states[index], modality_cells[index]),
                )
                active = current_valid[:, index].unsqueeze(-1)
                modality_states[index] = torch.where(
                    active, candidate_state, modality_states[index]
                )
                modality_cells[index] = torch.where(active, candidate_cell, modality_cells[index])

            nodes = torch.stack(modality_states, dim=1)
            if self.use_dynamic_graph:
                graph_nodes, messages = self.dynamic_graph(nodes, current_valid)
            else:
                graph_nodes = nodes * current_valid.unsqueeze(-1)
                messages = graph_nodes

            for index, cell in enumerate(self.view_memory_cells):
                active = current_valid[:, index].unsqueeze(-1)
                if self.use_view_memory:
                    candidate_view = cell(
                        torch.cat((graph_nodes[:, index], messages[:, index]), dim=-1),
                        view_memories[index],
                    )
                else:
                    candidate_view = graph_nodes[:, index]
                view_memories[index] = torch.where(active, candidate_view, view_memories[index])

            if self.use_fusion_memory:
                candidate_shared = self.fusion_memory(
                    torch.cat(view_memories, dim=-1), shared_memory
                )
            else:
                candidate_shared = torch.stack(view_memories, dim=1).mean(dim=1)
            frame_active = current_valid.any(dim=1, keepdim=True)
            shared_memory = torch.where(frame_active, candidate_shared, shared_memory)
            sequence_views.append(torch.cat((*view_memories, shared_memory), dim=-1))
            sequence_valid.append(frame_active.squeeze(-1))

        sequence = torch.stack(sequence_views, dim=1)
        valid_steps = torch.stack(sequence_valid, dim=1)
        if self.use_temporal_attention:
            attention_logits = self.temporal_attention(sequence).squeeze(-1)
            weights = _masked_softmax(attention_logits, valid_steps, dim=1).unsqueeze(-1)
            representation = (sequence * weights).sum(dim=1)
        else:
            weights = valid_steps.unsqueeze(-1).to(sequence.dtype)
            representation = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.classifier(representation)
