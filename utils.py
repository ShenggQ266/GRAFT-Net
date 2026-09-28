import math
import os
import random

import numpy as np
import torch

def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def linear_schedule(epoch: int, start_value: float, end_value: float, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return float(end_value)
    progress = min(max(epoch / float(ramp_epochs), 0.0), 1.0)
    return float(start_value + (end_value - start_value) * progress)


# Update the teacher model with an EMA of the student model.
def update_ema_variables(model, ema_model, global_step, total_steps, m_start=0.99, m_end=0.999):
    if total_steps <= 1:
        current_m = float(m_end)
    else:
        current_m = m_end - (m_end - m_start) * (math.cos(math.pi * global_step / (total_steps - 1)) + 1.0) / 2.0

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(current_m).add_(param.data, alpha=1.0 - current_m)

    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if torch.is_floating_point(buffer):
            ema_buffer.data.mul_(current_m).add_(buffer.data, alpha=1.0 - current_m)
        else:
            ema_buffer.data.copy_(buffer.data)

    return current_m


def get_infinite_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def safe_batch_size(requested_bs, n_samples):
    return max(1, min(int(requested_bs), int(n_samples)))


def safe_load_state_dict(model, ckpt_path, device):
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        return False
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return True


def build_or_load_split(cfg, n_total, save_dir):
    split_file = cfg.get("split_file", "")
    seed = int(cfg.get("seed", 2026))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    labeled_ratio = float(cfg.get("labeled_ratio", 0.2))

    if split_file and os.path.exists(split_file):
        print(f"[*] Loading split from existing file: {split_file}")
        saved_split = np.load(split_file, allow_pickle=False)
        required = ["idx_train_all", "idx_lb", "idx_ulb", "idx_val"]
        missing = [k for k in required if k not in saved_split.files]
        if missing:
            raise KeyError(f"Split file missing keys: {missing}")
        return saved_split, split_file

    if split_file:
        target_split_file = split_file
        split_dir = os.path.dirname(target_split_file)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
    else:
        target_split_file = os.path.join(save_dir, f"auto_split_seed{seed}_lb{labeled_ratio:.3f}_val{val_ratio:.3f}_test{test_ratio:.3f}.npz")

    if not (0.0 < labeled_ratio < 1.0):
        raise ValueError(f"labeled_ratio must be in (0,1), got {labeled_ratio}")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0):
        raise ValueError(f"Require 0 <= val_ratio,test_ratio < 1 and val_ratio+test_ratio < 1, got {val_ratio}, {test_ratio}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    max_reserved = max(0, n_total - 2)
    if n_test + n_val > max_reserved:
        overflow = n_test + n_val - max_reserved
        n_test = max(0, n_test - overflow)
    n_train_all = n_total - n_val - n_test
    if n_train_all <= 1:
        raise ValueError(f"Not enough samples after split: total={n_total}, val={n_val}, test={n_test}")

    idx_test = np.sort(perm[:n_test].astype(np.int64))
    idx_val = np.sort(perm[n_test:n_test+n_val].astype(np.int64))
    idx_train_all = np.sort(perm[n_test+n_val:].astype(np.int64))

    n_lb = int(round(n_train_all * labeled_ratio))
    n_lb = min(max(1, n_lb), n_train_all - 1)
    lb_sel = rng.permutation(idx_train_all)
    idx_lb = np.sort(lb_sel[:n_lb].astype(np.int64))
    idx_ulb = np.sort(lb_sel[n_lb:].astype(np.int64))

    np.savez(
        target_split_file,
        idx_train_all=idx_train_all,
        idx_lb=idx_lb,
        idx_ulb=idx_ulb,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    print(f"Split file not provided/found. Auto-generated split and saved to: {target_split_file}")
    saved_split = np.load(target_split_file, allow_pickle=False)
    return saved_split, target_split_file
