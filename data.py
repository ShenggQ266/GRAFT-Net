import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _TensorShapeView:
    n: int
    c: int
    h: int
    w: int

    @classmethod
    def from_4d(cls, x: torch.Tensor):
        n, c, h, w = x.shape
        return cls(int(n), int(c), int(h), int(w))


@dataclass(frozen=True)
class _ColumnPick:
    width: int
    ratio: float

    def count(self) -> int:
        wanted = max(1, int(round(self.width * self.ratio)))
        return min(wanted, self.width)

    def take(self) -> Sequence[int]:
        return random.sample(range(self.width), k=self.count())


@dataclass(frozen=True)
class _DropSpan:
    low: int
    high: int
    width: int

    def normalized(self) -> Tuple[int, int]:
        max_drop = max(1, min(int(self.high), self.width))
        min_drop = max(1, min(int(self.low), max_drop))
        return min_drop, max_drop

    def columns(self) -> Sequence[int]:
        min_drop, max_drop = self.normalized()
        num_drop = random.randint(min_drop, max_drop)
        return random.sample(range(self.width), k=min(num_drop, self.width))


@dataclass(frozen=True)
class _BlockWindow:
    height: int
    width: int
    ratio_h: float
    ratio_w: float

    def size(self) -> Tuple[int, int]:
        bh = max(1, int(self.height * self.ratio_h))
        bw = max(1, int(self.width * self.ratio_w))
        return bh, bw

    def origin(self) -> Tuple[int, int, int, int]:
        bh, bw = self.size()
        y0 = np.random.randint(0, max(1, self.height - bh + 1))
        x0 = np.random.randint(0, max(1, self.width - bw + 1))
        return y0, x0, bh, bw


class _TensorNoiseScale:
    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def signal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.std(x, dim=[1, 2, 3], keepdim=True, unbiased=False).clamp(min=1e-8)

    def snr(self) -> float:
        return 10.0 ** (self.snr_db / 20.0)

    def noise_std(self, x: torch.Tensor) -> torch.Tensor:
        return self.signal(x) / max(self.snr(), 1e-8)


class _TraceScaler:
    def __init__(self, scale_min: float, scale_max: float, max_ratio: float):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_ratio = max_ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        out = img.clone()
        shape = _TensorShapeView.from_4d(out)
        cols = _ColumnPick(shape.w, self.max_ratio).take()
        scales = torch.empty((len(cols),), device=out.device, dtype=out.dtype).uniform_(self.scale_min, self.scale_max)
        for i, c in enumerate(cols):
            out[:, :, :, c] *= scales[i]
        return out


class _Gate:
    def __init__(self, cfg: Mapping[str, Any], name: str):
        self.cfg = cfg
        self.name = name

    def on(self) -> bool:
        prob = self.cfg[self.name]
        return prob > 0.0 and random.random() < prob


class _WeakPolicy:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        if _Gate(self.cfg, "weak_noise_prob").on():
            snr_db = random.uniform(self.cfg["weak_snr_db_min"], self.cfg["weak_snr_db_max"])
            return add_awgn_noise(x, snr_db)
        return x

    def trace(self, x: torch.Tensor) -> torch.Tensor:
        if _Gate(self.cfg, "weak_trace_scale_prob").on():
            return apply_trace_scaling(x, self.cfg["weak_trace_scale_min"], self.cfg["weak_trace_scale_max"], max_ratio=0.05)
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out = self.noise(out)
        out = self.trace(out)
        return out


class _StrongPolicy:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        if _Gate(self.cfg, "strong_noise_prob").on():
            snr_db = random.uniform(self.cfg["strong_snr_db_min"], self.cfg["strong_snr_db_max"])
            return add_awgn_noise(x, snr_db)
        return x

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        if _Gate(self.cfg, "strong_scale_prob").on():
            return apply_trace_scaling(x, self.cfg["strong_scale_min"], self.cfg["strong_scale_max"], max_ratio=0.12)
        return x

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        shape = _TensorShapeView.from_4d(x)
        if _Gate(self.cfg, "strong_drop_prob").on():
            cols = _DropSpan(self.cfg["strong_drop_cols_min"], self.cfg["strong_drop_cols_max"], shape.w).columns()
            for c in cols:
                x[:, :, :, c] = 0.0
        return x

    def block(self, x: torch.Tensor) -> torch.Tensor:
        shape = _TensorShapeView.from_4d(x)
        if _Gate(self.cfg, "strong_block_prob").on():
            y0, x0, bh, bw = _BlockWindow(
                shape.h,
                shape.w,
                self.cfg["strong_block_ratio_h"],
                self.cfg["strong_block_ratio_w"],
            ).origin()
            x[:, :, y0:y0 + bh, x0:x0 + bw] = 0.0
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out = self.noise(out)
        out = self.scale(out)
        out = self.drop(out)
        out = self.block(out)
        return out


class _ArrayToMap:
    def __init__(self, src: Any):
        self.src = src

    def at(self, index: int) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(self.src[index], dtype=np.float32)).float()
        return _squeeze_to_2d(x).unsqueeze(0)


class _LabelToMap:
    def __init__(self, src: Optional[Any]):
        self.src = src

    def at(self, index: int) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(self.src[index], dtype=np.float32)).float()
        return _squeeze_to_2d(x).unsqueeze(0)


def _squeeze_to_2d(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def _as_real_index(indices: Any, local_index: int) -> Any:
    return indices[local_index]


def _is_labeled_mode(mode: str) -> bool:
    return mode in ["labeled", "val", "test"]


def add_awgn_noise(tensor, snr_db):
    meter = _TensorNoiseScale(snr_db)
    noise = torch.randn_like(tensor) * meter.noise_std(tensor)
    return tensor + noise


def apply_trace_scaling(img, scale_min, scale_max, max_ratio=0.08):
    return _TraceScaler(scale_min, scale_max, max_ratio)(img)


def gpr_weak_augmentation(img, cfg):
    return _WeakPolicy(cfg).run(img)


def gpr_strong_augmentation(img, cfg):
    return _StrongPolicy(cfg).run(img)


class GPR_Dataset(Dataset):
    def __init__(self, data, labels=None, indices=None, mode='labeled'):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.mode = mode
        self._data_bridge = _ArrayToMap(self.data)
        self._label_bridge = _LabelToMap(self.labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = _as_real_index(self.indices, idx)
        img = self._data_bridge.at(real_idx)
        if _is_labeled_mode(self.mode):
            target = self._label_bridge.at(real_idx)
            return img, target
        return img
