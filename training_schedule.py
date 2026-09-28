from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

try:
    from .utils import linear_schedule
except ImportError:
    from utils import linear_schedule


@dataclass(frozen=True)
class _ScheduleRange:
    start_key: str
    end_key: str
    ramp_key: str

    def value_at(self, epoch: int, cfg: Mapping[str, Any]) -> float:
        return linear_schedule(epoch, cfg[self.start_key], cfg[self.end_key], cfg[self.ramp_key])


@dataclass(frozen=True)
class _PhaseBase:
    high_keep: float
    mid_keep: float
    score_floor: float


@dataclass(frozen=True)
class _OptimShape:
    lr: float
    epochs_left: int

    @classmethod
    def from_args(cls, lr: float, total_epochs: int, start_epoch: int):
        left = max(1, int(total_epochs) - int(start_epoch))
        return cls(lr=lr, epochs_left=left)

    def bind(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs_left, eta_min=1e-6)
        return optimizer, scheduler


@dataclass
class _AnchorAugBuilder:
    cfg: Mapping[str, Any]

    def clone(self) -> Dict[str, Any]:
        return deepcopy(self.cfg)

    def prob_scale(self) -> float:
        return float(self.cfg["anchor_strong_prob_scale"])

    def mag_scale(self) -> float:
        return float(self.cfg["anchor_strong_mag_scale"])

    def apply_probabilities(self, aug: Dict[str, Any]) -> Dict[str, Any]:
        p_scale = self.prob_scale()
        prob_pairs = (
            ("strong_noise_prob", "strong_noise_prob"),
            ("strong_drop_prob", "strong_drop_prob"),
            ("strong_scale_prob", "strong_scale_prob"),
            ("strong_block_prob", "strong_block_prob"),
        )
        for target_key, source_key in prob_pairs:
            aug[target_key] = self.cfg[source_key] * p_scale
        return aug

    def apply_scale_window(self, aug: Dict[str, Any]) -> Dict[str, Any]:
        center = 1.0
        m_scale = self.mag_scale()
        aug["strong_scale_min"] = center - (center - self.cfg["strong_scale_min"]) * m_scale
        aug["strong_scale_max"] = center + (self.cfg["strong_scale_max"] - center) * m_scale
        return aug

    def apply_snr_window(self, aug: Dict[str, Any]) -> Dict[str, Any]:
        m_scale = self.mag_scale()
        snr_floor = self.cfg["strong_snr_db_min"] + (1.0 - m_scale) * 4.0
        snr_ceiling = self.cfg["strong_snr_db_max"] - (1.0 - m_scale) * 4.0
        aug["strong_snr_db_min"] = snr_floor
        aug["strong_snr_db_max"] = max(aug["strong_snr_db_min"] + 1.0, snr_ceiling)
        return aug

    def build(self) -> Dict[str, Any]:
        aug = self.clone()
        aug = self.apply_probabilities(aug)
        aug = self.apply_scale_window(aug)
        aug = self.apply_snr_window(aug)
        return aug


@dataclass(frozen=True)
class _PhaseSelector:
    epoch: int
    cfg: Mapping[str, Any]
    anchor_state: Mapping[str, Any]

    def scheduled_base(self) -> _PhaseBase:
        high = _ScheduleRange("high_keep_start", "high_keep_end", "high_keep_ramp_epochs").value_at(self.epoch, self.cfg)
        mid = _ScheduleRange("mid_keep_start", "mid_keep_end", "mid_keep_ramp_epochs").value_at(self.epoch, self.cfg)
        floor = _ScheduleRange("min_score_floor_start", "min_score_floor_end", "min_score_floor_ramp_epochs").value_at(self.epoch, self.cfg)
        return _PhaseBase(high_keep=high, mid_keep=mid, score_floor=floor)

    def is_anchor(self) -> bool:
        return bool(self.anchor_state["switched"])

    def is_explore(self) -> bool:
        return self.epoch <= int(self.cfg["exploration_end_epoch"])

    def anchor_values(self) -> Dict[str, Any]:
        return _phase_payload(
            phase="anchor",
            high_keep=float(self.cfg["anchor_high_keep"]),
            mid_keep=float(self.cfg["anchor_mid_keep"]),
            score_floor=float(self.cfg["anchor_score_floor"]),
            w_high=float(self.cfg["anchor_w_unsup_high"]),
            w_mid=float(self.cfg["anchor_w_unsup_mid"]),
            w_rank=float(self.cfg["anchor_w_unsup_rank"]),
            cap_ratio=float(self.cfg["anchor_unsup_cap_ratio"]),
            ema_start=float(self.cfg["anchor_ema_m_start"]),
            ema_end=float(self.cfg["anchor_ema_m_end"]),
            rank_enabled=not bool(self.cfg["anchor_disable_rank"]),
            aug_cfg=make_anchor_aug_cfg(self.cfg),
        )

    def explore_values(self, base: _PhaseBase) -> Dict[str, Any]:
        return _phase_payload(
            phase="explore",
            high_keep=base.high_keep,
            mid_keep=base.mid_keep,
            score_floor=base.score_floor,
            w_high=float(self.cfg["w_unsup_high"]),
            w_mid=float(self.cfg["w_unsup_mid"]),
            w_rank=float(self.cfg["w_unsup_rank"]),
            cap_ratio=float(self.cfg["unsup_cap_ratio"]),
            ema_start=float(self.cfg["ema_m_start"]),
            ema_end=float(self.cfg["ema_m_end"]),
            rank_enabled=True,
            aug_cfg=self.cfg,
        )

    def gain_values(self, base: _PhaseBase) -> Dict[str, Any]:
        return _phase_payload(
            phase="gain",
            high_keep=min(base.high_keep, float(self.cfg["gain_high_keep_cap"])),
            mid_keep=min(base.mid_keep, float(self.cfg["gain_mid_keep_cap"])),
            score_floor=max(base.score_floor, float(self.cfg["gain_score_floor_min"])),
            w_high=float(self.cfg["gain_w_unsup_high"]),
            w_mid=float(self.cfg["gain_w_unsup_mid"]),
            w_rank=float(self.cfg["gain_w_unsup_rank"]),
            cap_ratio=float(self.cfg["gain_unsup_cap_ratio"]),
            ema_start=float(self.cfg["ema_m_start"]),
            ema_end=float(self.cfg["ema_m_end"]),
            rank_enabled=True,
            aug_cfg=self.cfg,
        )

    def resolve(self) -> Dict[str, Any]:
        base = self.scheduled_base()
        if self.is_anchor():
            return self.anchor_values()
        if self.is_explore():
            return self.explore_values(base)
        return self.gain_values(base)


def _phase_payload(
    phase: str,
    high_keep: float,
    mid_keep: float,
    score_floor: float,
    w_high: float,
    w_mid: float,
    w_rank: float,
    cap_ratio: float,
    ema_start: float,
    ema_end: float,
    rank_enabled: bool,
    aug_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "phase": phase,
        "high_keep": high_keep,
        "mid_keep": mid_keep,
        "score_floor": score_floor,
        "w_unsup_high": w_high,
        "w_unsup_mid": w_mid,
        "w_unsup_rank": w_rank,
        "unsup_cap_ratio": cap_ratio,
        "ema_m_start": ema_start,
        "ema_m_end": ema_end,
        "rank_enabled": rank_enabled,
        "aug_cfg": aug_cfg,
    }
    return payload


@dataclass(frozen=True)
class _AnchorSwitchReadings:
    epoch: int
    best_mse: float
    val_mse: float
    avg_score: float
    avg_std: float
    avg_agree: float


@dataclass(frozen=True)
class _AnchorSwitchGate:
    cfg: Mapping[str, Any]
    anchor_state: Mapping[str, Any]
    readings: _AnchorSwitchReadings

    def already_switched(self) -> bool:
        return bool(self.anchor_state["switched"])

    def before_min_epoch(self) -> bool:
        return self.readings.epoch < int(self.cfg["anchor_min_epoch"])

    def plateau(self) -> bool:
        return self.anchor_state["epochs_since_best"] >= int(self.cfg["anchor_patience"])

    def overconfident(self) -> bool:
        metric_gap = self.readings.val_mse > self.readings.best_mse * (1.0 + float(self.cfg["anchor_rel_gap"]))
        score_ok = self.readings.avg_score >= float(self.cfg["anchor_score_thr"])
        std_ok = self.readings.avg_std <= float(self.cfg["anchor_std_thr"])
        agree_ok = self.readings.avg_agree >= float(self.cfg["anchor_agree_thr"])
        return metric_gap and score_ok and std_ok and agree_ok

    def decision(self) -> Tuple[bool, str]:
        if self.already_switched():
            return False, ""
        if self.before_min_epoch():
            return False, ""

        plateau = self.plateau()
        overconf = self.overconfident()

        if plateau and overconf:
            return True, "plateau+overconf"
        if plateau:
            return True, "plateau"
        if overconf:
            return True, "overconf"
        return False, ""


def build_optimizer_and_scheduler(model, lr, total_epochs, start_epoch=0):
    return _OptimShape.from_args(lr, total_epochs, start_epoch).bind(model)


def make_anchor_aug_cfg(cfg):
    return _AnchorAugBuilder(cfg).build()


def get_phase_params(epoch, cfg, anchor_state):
    return _PhaseSelector(epoch=epoch, cfg=cfg, anchor_state=anchor_state).resolve()


def should_switch_to_anchor(epoch, cfg, anchor_state, best_mse, val_mse, avg_score, avg_std, avg_agree):
    readings = _AnchorSwitchReadings(
        epoch=epoch,
        best_mse=best_mse,
        val_mse=val_mse,
        avg_score=avg_score,
        avg_std=avg_std,
        avg_agree=avg_agree,
    )
    return _AnchorSwitchGate(cfg=cfg, anchor_state=anchor_state, readings=readings).decision()
