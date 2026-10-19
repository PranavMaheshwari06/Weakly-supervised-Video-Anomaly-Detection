from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HierMilOutput:
    bag_embedding: torch.Tensor      # [B, D]
    attention: torch.Tensor          # [B, M]
    binary_logits: torch.Tensor      # [B, 2]   -> 0=Normal, 1=Anomaly
    anomaly_logits: torch.Tensor     # [B, K]   -> anomaly-only classes


class HierarchicalAttentionMilHead(nn.Module):
    """
    Shared attention encoder with Top-K routing:
    1) Binary head — Normal vs Anomaly
    2) Anomaly head — K-way classifier over top-attended clips
    """
    def __init__(
        self,
        embedding_dim: int,
        num_anomaly_classes: int,
        attention_hidden_dim: int = 256,
        dropout: float = 0.1,
        top_k_ratio: float = 0.25, # Route top 25% of clips to the anomaly head
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_anomaly_classes = int(num_anomaly_classes)
        self.top_k_ratio = top_k_ratio

        self.attention_fc = nn.Linear(self.embedding_dim, attention_hidden_dim)
        self.attention_score = nn.Linear(attention_hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        
        # Binary head uses standard attention pooling (1x embedding_dim)
        self.binary_classifier = nn.Linear(self.embedding_dim, 2)
        
        # Anomaly head uses Multi-Pooling (Attn + Max + Mean) -> (3x embedding_dim)
        self.anomaly_classifier = nn.Linear(self.embedding_dim * 3, self.num_anomaly_classes)

    def encode_bag(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 3
        B, M, D = x.shape

        h = torch.tanh(self.attention_fc(x))
        attn_logits = self.attention_score(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=1)

        z_binary = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        z_binary = self.dropout(z_binary)

        # route top-K attended clips to the anomaly classifier
        k_count = max(1, int(M * self.top_k_ratio))
        _, topk_idx = torch.topk(attn, k=k_count, dim=1)

        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        topk_x = torch.gather(x, 1, topk_idx_expanded)

        z_max, _ = torch.max(topk_x, dim=1)
        z_mean = torch.mean(topk_x, dim=1)

        z_anomaly_fused = torch.cat([z_binary, z_max, z_mean], dim=1)
        z_anomaly_fused = self.dropout(z_anomaly_fused)

        return z_binary, z_anomaly_fused, attn

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> HierMilOutput:
        z_binary, z_anomaly_fused, attention = self.encode_bag(x, mask)
        
        binary_logits = self.binary_classifier(z_binary)             # [B,2]
        anomaly_logits = self.anomaly_classifier(z_anomaly_fused)    # [B,K_classes]

        return HierMilOutput(
            bag_embedding=z_binary, # Pass back base for consistency
            attention=attention,
            binary_logits=binary_logits,
            anomaly_logits=anomaly_logits,
        )

    @staticmethod
    def compose_full_probs(binary_logits: torch.Tensor, anomaly_logits: torch.Tensor) -> torch.Tensor:
        p_bin = torch.softmax(binary_logits, dim=1)          # [B,2]
        p_anom = torch.softmax(anomaly_logits, dim=1)        # [B,K]

        p_normal = p_bin[:, 0:1]                             # [B,1]
        p_anomaly_classes = p_bin[:, 1:2] * p_anom           # [B,K]
        full_probs = torch.cat([p_normal, p_anomaly_classes], dim=1)
        return full_probs