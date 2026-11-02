"""Reward model: cosine VQ persona prototype scored against two responses."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from epipersona.paths import repo_root

_THIRD_PARTY = repo_root() / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


def _mlp(in_dim: int, out_dim: int, activation) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation(),
        nn.Dropout(0.1),
        nn.Linear(out_dim, out_dim),
    )


class CodebookMonitor:
    def __init__(self, codebook_size: int):
        self.codebook_size = codebook_size
        self.enabled = os.environ.get("EPIPERSONA_DEBUG", "") == "1"
        self.log_every = int(os.environ.get("CODEBOOK_LOG_EVERY", "150"))
        self._step = 0
        self._idx_buf = []
        self._commit_buf = []
        self._div_buf = []

    def update(self, indices, commit_loss, diversity_loss) -> None:
        if not self.enabled:
            return
        self._step += 1
        flat = indices.detach().reshape(-1)
        valid = flat[flat >= 0].long().cpu()
        self._idx_buf.append(valid)
        self._commit_buf.append(float(commit_loss))
        self._div_buf.append(float(diversity_loss))
        if self._step % self.log_every == 0 and _rank() == 0:
            self._report()
            self._idx_buf.clear()
            self._commit_buf.clear()
            self._div_buf.clear()

    def _report(self) -> None:
        if not self._idx_buf:
            return
        counts = torch.bincount(torch.cat(self._idx_buf), minlength=self.codebook_size)
        used = int((counts > 0).sum())
        probs = counts.float() / counts.sum().clamp(min=1)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        perplexity = math.exp(min(entropy, 700))
        print(
            f"codebook step={self._step} used={used}/{self.codebook_size} "
            f"perplexity={perplexity:.2f} entropy={entropy:.4f}"
        )


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class SoftPromptGenerator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        codebook_size: int,
        codebook_dim: int,
        diversity_temp: float = 0.5,
        hard_usage_weight: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.diversity_temp = diversity_temp
        self.hard_usage_weight = hard_usage_weight
        self.mlp1 = _mlp(hidden_size, hidden_size, nn.GELU)
        self.mlp2 = _mlp(hidden_size, hidden_size, nn.ReLU)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.vq = VectorQuantize(
            dim=hidden_size,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            use_cosine_sim=True,
            threshold_ema_dead_code=1,
            decay=0.5,
            commitment_weight=2.0,
            kmeans_init=True,
            kmeans_iters=20,
        )
        self.monitor = CodebookMonitor(codebook_size)

    def _diversity_loss(self, pre_quantized: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat = pre_quantized.reshape(-1, pre_quantized.size(-1))
        project_in = getattr(self.vq, "project_in", None)
        projected = project_in(flat) if project_in is not None else flat
        valid = projected[mask.reshape(-1).bool()]
        if valid.size(0) == 0:
            return torch.zeros((), device=pre_quantized.device)
        codebook = self.vq._codebook.embed.detach()
        if codebook.dim() == 3:
            codebook = codebook.squeeze(0)
        similarity = torch.matmul(
            F.normalize(valid, dim=-1),
            F.normalize(codebook, dim=-1).t(),
        ) / self.diversity_temp
        soft_assign = F.softmax(similarity, dim=-1)
        average = soft_assign.mean(0)
        soft_div = (average * (average + 1e-8).log()).sum()
        if self.hard_usage_weight == 0:
            return soft_div
        hard = F.one_hot(similarity.argmax(dim=-1), num_classes=codebook.size(0)).to(soft_assign.dtype)
        straight = hard + soft_assign - soft_assign.detach()
        average_hard = straight.mean(0)
        hard_div = (average_hard * (average_hard + 1e-8).log()).sum()
        return soft_div + self.hard_usage_weight * hard_div

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        batch, length, hidden = embeddings.shape
        mapped = self.mlp1(embeddings.reshape(-1, hidden)).view(batch, length, -1)
        quantized, indices, commit_loss = self.vq(mapped, mask=mask.bool())
        diversity_loss = self._diversity_loss(mapped, mask)
        self.monitor.update(indices, commit_loss, -diversity_loss)
        return quantized, indices, commit_loss, diversity_loss


class RewardModel(nn.Module):
    def __init__(
        self,
        base_model,
        codebook_size: int,
        codebook_dim: int,
        diversity_temp: float = 0.5,
        hard_usage_weight: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.soft_prompt_generator = SoftPromptGenerator(
            hidden_size=self.hidden_size,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            diversity_temp=diversity_temp,
            hard_usage_weight=hard_usage_weight,
        )
        self.reward_norm = nn.LayerNorm(self.hidden_size)
        self.score_head = nn.Linear(self.hidden_size, 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.base_model.get_input_embeddings()(input_ids)

    def _score(self, input_ids1, mask1, input_ids2, mask2, quantized, use_cache, **kwargs):
        batch = input_ids1.size(0)
        combined_emb = torch.cat([self._embed(input_ids1), self._embed(input_ids2)], dim=0)
        combined_mask = torch.cat([mask1, mask2], dim=0)
        outputs = self.base_model(
            inputs_embeds=combined_emb,
            attention_mask=combined_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        mask = combined_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        first, second = pooled[:batch], pooled[batch:]
        prototype = quantized.mean(dim=1)
        score1 = (F.normalize(first, dim=-1) * F.normalize(prototype, dim=-1)).sum(-1) / 0.1
        score2 = (F.normalize(second, dim=-1) * F.normalize(prototype, dim=-1)).sum(-1) / 0.1
        loss = -F.log_softmax(torch.stack([score1, score2], dim=-1), dim=-1)[:, 0].mean()
        probs = torch.sigmoid(score1 - score2)
        return loss, probs, score1, score2

    def forward(
        self,
        input_ids,
        attention_mask,
        resp2_input_ids,
        resp2_attention_mask,
        allocated_embedding_stat,
        allocated_embedding_stat_mask,
        labels=None,
        use_cache=False,
        **kwargs,
    ):
        del labels
        quantized, indices, commit_loss, diversity_loss = self.soft_prompt_generator(
            allocated_embedding_stat, allocated_embedding_stat_mask
        )
        loss, probs, score1, score2 = self._score(
            input_ids,
            attention_mask,
            resp2_input_ids,
            resp2_attention_mask,
            quantized,
            use_cache,
        )
        return {
            "loss": loss,
            "logits": probs.detach(),
            "score_resp1": score1,
            "score_resp2": score2,
            "vq_commit_loss": commit_loss,
            "vq_diversity_loss": diversity_loss,
            "vq_indices": indices,
        }
