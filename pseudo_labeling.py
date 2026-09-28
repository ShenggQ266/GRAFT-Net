import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _TeacherViewPair:
    first: torch.Tensor
    second: torch.Tensor

    def mean_probs(self) -> torch.Tensor:
        return 0.5 * (self.first + self.second)

    def mean_maps(self, dm) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_1, _ = dm.get_mean_std_from_probs(self.first)
        mu_2, _ = dm.get_mean_std_from_probs(self.second)
        return mu_1, mu_2


@dataclass(frozen=True)
class _ScoreTerms:
    mu: torch.Tensor
    std: torch.Tensor
    max_prob: torch.Tensor
    disagreement: torch.Tensor
    std_conf: torch.Tensor
    agree_conf: torch.Tensor
    score: torch.Tensor

    def as_parts(self) -> Dict[str, torch.Tensor]:
        return {
            "mu": self.mu.detach(),
            "std": self.std.detach(),
            "max_prob": self.max_prob.detach(),
            "disagreement": self.disagreement.detach(),
            "std_conf": self.std_conf.detach(),
            "agree_conf": self.agree_conf.detach(),
            "score": self.score.detach(),
        }


@dataclass(frozen=True)
class _KeepResult:
    mask: torch.Tensor
    threshold: float
    keep_actual: float


class _ReliabilityMixer:
    def __init__(self, dm, gamma: float, agreement_scale: float, std_floor: float):
        self.dm = dm
        self.gamma = float(gamma)
        self.agreement_scale = float(max(agreement_scale, 1e-6))
        self.std_floor = float(max(std_floor, 0.0))

    def score_parts(self, probs_avg: torch.Tensor, mu_1: torch.Tensor, mu_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_avg, std_avg = self.dm.get_mean_std_from_probs(probs_avg)
        max_prob = probs_avg.max(dim=1, keepdim=True).values
        disagreement = self._agreement_gap(mu_1, mu_2)
        std_conf = self._std_reliability(std_avg)
        agree_conf = self._agreement_reliability(disagreement)
        score = (std_conf * agree_conf * max_prob).detach()
        return _ScoreTerms(
            mu=mu_avg,
            std=std_avg,
            max_prob=max_prob,
            disagreement=disagreement,
            std_conf=std_conf,
            agree_conf=agree_conf,
            score=score,
        ).as_parts()

    def _agreement_gap(self, mu_1: torch.Tensor, mu_2: torch.Tensor) -> torch.Tensor:
        return (mu_1 - mu_2).abs() / self.dm.value_range

    def _std_reliability(self, std_avg: torch.Tensor) -> torch.Tensor:
        std_conf = (1.0 - std_avg / max(self.dm.std_max, 1e-6)).clamp(min=self.std_floor, max=1.0)
        return std_conf.pow(self.gamma)

    def _agreement_reliability(self, disagreement: torch.Tensor) -> torch.Tensor:
        return (1.0 - disagreement / self.agreement_scale).clamp(0.0, 1.0)


class _TopkGate:
    def __init__(self, score: torch.Tensor):
        self.score = score
        self.flat = score.view(-1)
        self.total = self.flat.numel()

    def select(self, keep_ratio: float, min_score_floor: float) -> _KeepResult:
        keep_ratio = float(min(max(keep_ratio, 0.0), 1.0))
        keep_count = self._keep_count(keep_ratio)
        if keep_count <= 0:
            return _KeepResult(torch.zeros_like(self.score), 0.0, 0.0)
        threshold = self._threshold_at(keep_count, min_score_floor)
        mask = (self.score >= threshold).float()
        keep_actual = float(mask.mean().item())
        return _KeepResult(mask, threshold, keep_actual)

    def _keep_count(self, keep_ratio: float) -> int:
        keep_count = max(1, int(math.ceil(self.total * keep_ratio)))
        return min(keep_count, self.total)

    def _threshold_at(self, keep_count: int, min_score_floor: float) -> float:
        topk_vals = torch.topk(self.flat, k=keep_count, largest=True, sorted=True).values
        threshold = float(topk_vals[-1].item())
        return max(threshold, float(min_score_floor))


class _RegionWeights:
    def __init__(self, score: torch.Tensor, high: _KeepResult, mid: _KeepResult):
        self.score = score
        self.high = high
        self.mid = mid

    def mid_only(self) -> torch.Tensor:
        return (self.mid.mask - self.high.mask).clamp(min=0.0, max=1.0)

    def high_weight(self) -> torch.Tensor:
        return self.high.mask * self.score

    def mid_weight(self) -> torch.Tensor:
        return self.mid_only() * self.score


class _PseudoStats:
    def __init__(self, parts: Dict[str, torch.Tensor], high: _KeepResult, mid: _KeepResult):
        self.parts = parts
        self.high = high
        self.mid = mid

    def as_dict(self) -> Dict[str, float]:
        return {
            "high_thr": self.high.threshold,
            "mid_thr": self.mid.threshold,
            "keep_high": self.high.keep_actual,
            "keep_mid": self.mid.keep_actual,
            "score_mean": float(self.parts["score"].mean().item()),
            "std_mean": float(self.parts["std"].mean().item()),
            "agree_mean": float(self.parts["agree_conf"].mean().item()),
            "prob_mean": float(self.parts["max_prob"].mean().item()),
        }


class AdaptiveThresholdPseudoLabeler(nn.Module):
    def __init__(self, dist_manager, reliability_gamma=2.0, agreement_scale=0.12, std_floor=0.0):
        super().__init__()
        self.dm = dist_manager
        self.reliability_gamma = float(reliability_gamma)
        self.agreement_scale = float(max(agreement_scale, 1e-6))
        self.std_floor = float(max(std_floor, 0.0))
        self._score_mixer = _ReliabilityMixer(
            self.dm,
            self.reliability_gamma,
            self.agreement_scale,
            self.std_floor,
        )

    def _build_score(self, probs_avg, mu_1, mu_2):
        return self._score_mixer.score_parts(probs_avg, mu_1, mu_2)

    @staticmethod
    def _topk_mask(score, keep_ratio, min_score_floor):
        picked = _TopkGate(score).select(keep_ratio, min_score_floor)
        return picked.mask, picked.threshold, picked.keep_actual

    def _score_from_teachers(self, teacher_probs_1, teacher_probs_2):
        pair = _TeacherViewPair(teacher_probs_1, teacher_probs_2)
        probs_avg = pair.mean_probs()
        mu_1, mu_2 = pair.mean_maps(self.dm)
        parts = self._build_score(probs_avg, mu_1, mu_2)
        return probs_avg, parts

    def _make_regions(self, score, high_keep_ratio, mid_keep_ratio, min_score_floor):
        gate = _TopkGate(score)
        high = gate.select(high_keep_ratio, min_score_floor)
        mid = gate.select(mid_keep_ratio, min_score_floor)
        weights = _RegionWeights(score, high, mid)
        return high, mid, weights

    def _pack_forward_result(self, probs_avg, parts, high, mid, weights):
        return {
            "teacher_mu": parts["mu"],
            "teacher_probs": probs_avg.detach(),
            "teacher_score": parts["score"],
            "teacher_std": parts["std"],
            "high_mask": high.mask,
            "mid_mask": weights.mid_only(),
            "high_weight": weights.high_weight(),
            "mid_weight": weights.mid_weight(),
            "stats": _PseudoStats(parts, high, mid).as_dict(),
        }

    def forward(self, teacher_probs_1, teacher_probs_2, high_keep_ratio, mid_keep_ratio, min_score_floor):
        probs_avg, parts = self._score_from_teachers(teacher_probs_1, teacher_probs_2)
        high, mid, weights = self._make_regions(
            parts["score"],
            high_keep_ratio,
            mid_keep_ratio,
            min_score_floor,
        )
        return self._pack_forward_result(probs_avg, parts, high, mid, weights)
