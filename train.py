import json
import os
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from BLNet import WaveletMambaBinningUNet
except ImportError:
    try:
        from .BLNet import WaveletMambaBinningUNet
    except ImportError:
        print("Error: BLNet.py not found. Please ensure it is in the Python path or the current directory.")
        raise

try:
    from .config import CONFIG
    from .data import GPR_Dataset, gpr_strong_augmentation, gpr_weak_augmentation
    from .distribution_losses import LinearDistributionManager, pairwise_rank_loss_from_maps, ssim_score, masked_smooth_l1
    from .evaluation import evaluate_on_loader
    from .pseudo_labeling import AdaptiveThresholdPseudoLabeler
    from .training_schedule import build_optimizer_and_scheduler, get_phase_params, should_switch_to_anchor
    from .utils import (
        build_or_load_split,
        get_infinite_iterator,
        safe_batch_size,
        safe_load_state_dict,
        set_seed,
        update_ema_variables,
    )
except ImportError:
    from config import CONFIG
    from data import GPR_Dataset, gpr_strong_augmentation, gpr_weak_augmentation
    from distribution_losses import LinearDistributionManager, pairwise_rank_loss_from_maps, ssim_score, masked_smooth_l1
    from evaluation import evaluate_on_loader
    from pseudo_labeling import AdaptiveThresholdPseudoLabeler
    from training_schedule import build_optimizer_and_scheduler, get_phase_params, should_switch_to_anchor
    from utils import (
        build_or_load_split,
        get_infinite_iterator,
        safe_batch_size,
        safe_load_state_dict,
        set_seed,
        update_ema_variables,
    )


@dataclass
class SplitPack:
    saved_split: object
    used_split_file: str
    idx_train_all: object
    idx_lb: object
    idx_ulb: object
    idx_val: object
    idx_test: object


@dataclass
class DataPack:
    full_npz: object
    full_data: object
    full_labels: object
    split: SplitPack
    phys_min: float
    phys_max: float


@dataclass
class ModelPack:
    model: nn.Module
    ema_model: nn.Module
    optimizer: object
    scheduler: object
    dist_manager: object
    pseudo_labeler: object


@dataclass
class TrainState:
    best_mse: float
    global_step: int
    anchor_state: dict


_FIXED_TRAINING_CFG = {'num_bins': 34,
 'soft_sigma': 1.0,
 'w_ldl': 0.1,
 'w_mse': 1.0,
 'w_ssim': 5.0,
 'ema_m_start': 0.99,
 'ema_m_end': 0.999,
 'w_unsup_high': 0.42,
 'w_unsup_mid': 0.12,
 'w_unsup_rank': 0.1,
 'unsup_cap_ratio': 0.65,
 'reliability_gamma': 2.0,
 'high_keep_start': 0.1,
 'high_keep_end': 0.32,
 'high_keep_ramp_epochs': 35,
 'mid_keep_start': 0.24,
 'mid_keep_end': 0.6,
 'mid_keep_ramp_epochs': 45,
 'agreement_scale': 0.14,
 'std_floor': 0.0,
 'min_score_floor_start': 0.16,
 'min_score_floor_end': 0.06,
 'min_score_floor_ramp_epochs': 35,
 'exploration_end_epoch': 18,
 'gain_high_keep_cap': 0.28,
 'gain_mid_keep_cap': 0.46,
 'gain_score_floor_min': 0.08,
 'gain_w_unsup_high': 0.4,
 'gain_w_unsup_mid': 0.07,
 'gain_w_unsup_rank': 0.05,
 'gain_unsup_cap_ratio': 0.55,
 'anchor_min_epoch': 36,
 'anchor_patience': 4,
 'anchor_rel_gap': 0.02,
 'anchor_score_thr': 0.34,
 'anchor_std_thr': 1.25,
 'anchor_agree_thr': 0.88,
 'anchor_reload_best': True,
 'anchor_lr_scale': 0.35,
 'anchor_high_keep': 0.2,
 'anchor_mid_keep': 0.28,
 'anchor_score_floor': 0.1,
 'anchor_w_unsup_high': 0.26,
 'anchor_w_unsup_mid': 0.03,
 'anchor_w_unsup_rank': 0.0,
 'anchor_unsup_cap_ratio': 0.38,
 'anchor_ema_m_start': 0.9992,
 'anchor_ema_m_end': 0.9995,
 'anchor_disable_rank': True,
 'anchor_strong_prob_scale': 0.65,
 'anchor_strong_mag_scale': 0.6,
 'weak_noise_prob': 0.35,
 'weak_snr_db_min': 0.0,
 'weak_snr_db_max': 5.0,
 'weak_trace_scale_prob': 0.15,
 'weak_trace_scale_min': 0.98,
 'weak_trace_scale_max': 1.02,
 'strong_noise_prob': 0.5,
 'strong_snr_db_min': 8.0,
 'strong_snr_db_max': 20.0,
 'strong_drop_prob': 0.35,
 'strong_drop_cols_min': 1,
 'strong_drop_cols_max': 2,
 'strong_scale_prob': 0.4,
 'strong_scale_min': 0.9,
 'strong_scale_max': 1.1,
 'strong_block_prob': 0.0,
 'strong_block_ratio_h': 0.08,
 'strong_block_ratio_w': 0.08,
 'coarse_pool_kernel': 4,
 'rank_pool_kernel': 8,
 'rank_min_valid_ratio': 0.6,
 'rank_pairs_per_image': 96,
 'rank_teacher_margin': 0.28,
 'rank_temperature': 0.65,
 'rank_weight_power': 1.0,
 'save_teacher_name': 'best_model_teacher.pth',
 'save_student_name': 'best_model_student.pth'}

_RUNTIME_CFG_DEFAULTS = {'data_path': '',
 'save_dir': './ssl_outputs',
 'split_file': '',
 'pretrained_path': '',
 'val_ratio': 0.1,
 'test_ratio': 0.1,
 'labeled_ratio': 0.2,
 'seed': 2026,
 'epochs': 150,
 'lr': 0.0001,
 'num_workers': 4,
 'gpu': 0,
 'bs_lb': 4,
 'bs_ulb': 4,
 'bs_eval': 8,
 'phys_min': None,
 'phys_max': None}

_RUNTIME_OVERRIDE_KEYS = tuple(_RUNTIME_CFG_DEFAULTS.keys())


def _as_plain_dict(args):
    if isinstance(args, SimpleNamespace):
        return vars(args)
    if args is None:
        return {}
    return dict(args)


def _normalize_runtime_cfg(cfg):
    # Avoid the original placeholder value " " being treated as a valid path.
    for key in ("data_path", "save_dir", "split_file", "pretrained_path"):
        if isinstance(cfg.get(key), str):
            cfg[key] = cfg[key].strip()
    if not cfg["save_dir"]:
        cfg["save_dir"] = _RUNTIME_CFG_DEFAULTS["save_dir"]
    return cfg


def _validate_runtime_cfg(cfg):
    if not cfg["data_path"]:
        raise ValueError("Please set CONFIG['data_path'] to a valid .npz file path before training.")
    if not os.path.isfile(cfg["data_path"]):
        raise FileNotFoundError(f"CONFIG['data_path'] does not exist: {cfg['data_path']}")
    if cfg["split_file"] and not os.path.isfile(cfg["split_file"]):
        print(f"[Warning] split_file not found, a new split may be created: {cfg['split_file']}")
    return cfg


def prepare_cfg(args):
    user_cfg = _as_plain_dict(args)
    cfg = deepcopy(_FIXED_TRAINING_CFG)
    cfg.update(deepcopy(_RUNTIME_CFG_DEFAULTS))
    for key in _RUNTIME_OVERRIDE_KEYS:
        if key in user_cfg:
            cfg[key] = user_cfg[key]
    cfg = _normalize_runtime_cfg(cfg)
    _validate_runtime_cfg(cfg)
    return cfg


def _printable_runtime_cfg(cfg):
    return {key: cfg[key] for key in _RUNTIME_OVERRIDE_KEYS}


def _printable_fixed_summary(cfg):
    return {
        "num_bins": cfg["num_bins"],
        "loss_weights": {
            "w_mse": cfg["w_mse"],
            "w_ssim": cfg["w_ssim"],
            "w_ldl": cfg["w_ldl"],
        },
        "ssl_weights": {
            "w_unsup_high": cfg["w_unsup_high"],
            "w_unsup_mid": cfg["w_unsup_mid"],
            "w_unsup_rank": cfg["w_unsup_rank"],
            "unsup_cap_ratio": cfg["unsup_cap_ratio"],
        },
        "rank": {
            "pool_kernel": cfg["rank_pool_kernel"],
            "pairs_per_image": cfg["rank_pairs_per_image"],
        },
        "anchor": {
            "min_epoch": cfg["anchor_min_epoch"],
            "patience": cfg["anchor_patience"],
            "reload_best": cfg["anchor_reload_best"],
        },
    }


def get_train_device(cfg):
    device_name = f'cuda:{cfg["gpu"]}' if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def print_run_header(cfg):
    print("=" * 100)
    print("Adaptive-threshold SSL config: runtime config + fixed internal training parameters")
    print("[Runtime CONFIG]")
    print(json.dumps(_printable_runtime_cfg(cfg), indent=2, ensure_ascii=False))
    print("[Fixed training summary]")
    print(json.dumps(_printable_fixed_summary(cfg), indent=2, ensure_ascii=False))
    print("=" * 100)


def load_npz_fields(cfg):
    print(f"[*] Loading data from {cfg['data_path']}...")
    full_npz = np.load(cfg["data_path"], mmap_mode="r")
    keys = set(full_npz.files)
    if "data" not in keys or "labels" not in keys:
        raise KeyError(f"NPZ file must contain 'data' and 'labels', got {sorted(keys)}")
    return full_npz, full_npz["data"], full_npz["labels"]


def fetch_split_pack(cfg, full_data):
    saved_split, used_split_file = build_or_load_split(cfg, n_total=len(full_data), save_dir=cfg["save_dir"])
    cfg["split_file_used"] = used_split_file
    idx_test = saved_split["idx_test"] if "idx_test" in saved_split.files else None
    return SplitPack(
        saved_split=saved_split,
        used_split_file=used_split_file,
        idx_train_all=saved_split["idx_train_all"],
        idx_lb=saved_split["idx_lb"],
        idx_ulb=saved_split["idx_ulb"],
        idx_val=saved_split["idx_val"],
        idx_test=idx_test,
    )


def decide_phys_range(cfg, full_labels, idx_train_all):
    if cfg["phys_min"] is None:
        phys_min = float(np.min(np.asarray(full_labels[idx_train_all], dtype=np.float32)))
    else:
        phys_min = float(cfg["phys_min"])
    if cfg["phys_max"] is None:
        phys_max = float(np.max(np.asarray(full_labels[idx_train_all], dtype=np.float32)))
    else:
        phys_max = float(cfg["phys_max"])
    if not np.isfinite(phys_min) or not np.isfinite(phys_max):
        raise ValueError(f"Invalid physical range: {phys_min}, {phys_max}")
    return phys_min, phys_max


def collect_data_pack(cfg):
    full_npz, full_data, full_labels = load_npz_fields(cfg)
    split = fetch_split_pack(cfg, full_data)
    phys_min, phys_max = decide_phys_range(cfg, full_labels, split.idx_train_all)
    return DataPack(full_npz, full_data, full_labels, split, phys_min, phys_max)


def make_wmb(cfg, phys_min, phys_max, device):
    return WaveletMambaBinningUNet(num_bins=cfg["num_bins"], min_val=phys_min, max_val=phys_max).to(device)


def load_student_start(model, cfg, device):
    loaded = safe_load_state_dict(model, cfg["pretrained_path"], device)
    if loaded:
        print(f"Loaded pretrained weights from {cfg['pretrained_path']}")
    else:
        print("Pretrained file not found or empty path. Start SSL from scratch.")
    return loaded


def align_teacher_to_student(model, ema_model):
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False


def prepare_model_pack(cfg, data_pack, device):
    model = make_wmb(cfg, data_pack.phys_min, data_pack.phys_max, device)
    ema_model = make_wmb(cfg, data_pack.phys_min, data_pack.phys_max, device)
    load_student_start(model, cfg, device)
    align_teacher_to_student(model, ema_model)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg["lr"], cfg["epochs"], start_epoch=0)
    dist_manager = LinearDistributionManager(num_bins=cfg["num_bins"], min_val=data_pack.phys_min, max_val=data_pack.phys_max, device=device)
    pseudo_labeler = AdaptiveThresholdPseudoLabeler(
        dist_manager=dist_manager,
        reliability_gamma=cfg["reliability_gamma"],
        agreement_scale=cfg["agreement_scale"],
        std_floor=cfg["std_floor"],
    ).to(device)
    return ModelPack(model, ema_model, optimizer, scheduler, dist_manager, pseudo_labeler)


def initial_train_state():
    return TrainState(
        best_mse=float("inf"),
        global_step=0,
        anchor_state={
            "switched": False,
            "switch_epoch": -1,
            "epochs_since_best": 0,
            "switch_reason": "",
        },
    )


def meter_template():
    return {
        "sup_total": 0.0, "sup_mse": 0.0, "sup_ssim": 0.0, "sup_ldl": 0.0,
        "uns_high": 0.0, "uns_mid": 0.0, "uns_rank": 0.0, "uns_scale": 0.0,
        "rank_pairs": 0.0, "keep_high": 0.0, "keep_mid": 0.0,
        "score_mean": 0.0, "std_mean": 0.0, "agree_mean": 0.0, "ema_m": 0.0,
    }


def take_batch_pair(iter_lb, iter_ulb, device):
    img_x, target_phys = next(iter_lb)
    img_u = next(iter_ulb)
    img_x = img_x.to(device, non_blocking=True)
    target_phys = target_phys.to(device, non_blocking=True)
    img_u = img_u.to(device, non_blocking=True)
    return img_x, target_phys, img_u


def resize_logits_if_needed(logits_x, target_phys):
    if logits_x.shape[2:] != target_phys.shape[2:]:
        logits_x = F.interpolate(logits_x, size=target_phys.shape[2:], mode="bilinear", align_corners=False)
    return logits_x


def labeled_branch(model, dist_manager, img_x, target_phys, cfg, phys_min, phys_max):
    logits_x, _, _ = model(img_x)
    logits_x = resize_logits_if_needed(logits_x, target_phys)
    pred_x_phys = dist_manager.get_val(logits_x)
    loss_mse = F.mse_loss(pred_x_phys, target_phys)
    loss_ssim = 1.0 - ssim_score(pred_x_phys, target_phys, data_range=max(phys_max - phys_min, 1e-6))
    target_soft = dist_manager.generate_gaussian_soft_labels(target_phys, sigma=cfg["soft_sigma"])
    loss_ldl = F.kl_div(F.log_softmax(logits_x, dim=1), target_soft, reduction="none").sum(dim=1).mean()
    loss_sup = cfg["w_mse"] * loss_mse + cfg["w_ssim"] * loss_ssim + cfg["w_ldl"] * loss_ldl
    return {
        "loss_sup": loss_sup,
        "loss_mse": loss_mse,
        "loss_ssim": loss_ssim,
        "loss_ldl": loss_ldl,
    }


def make_augmented_unlabeled(img_u, aug_cfg):
    img_u_w1 = gpr_weak_augmentation(img_u, aug_cfg)
    img_u_w2 = gpr_weak_augmentation(img_u, aug_cfg)
    img_u_s = gpr_strong_augmentation(img_u, aug_cfg)
    return img_u_w1, img_u_w2, img_u_s


def teacher_pseudo_branch(ema_model, dist_manager, pseudo_labeler, img_u_w1, img_u_w2, high_keep, mid_keep, score_floor):
    with torch.no_grad():
        logits_t_w1, _, _ = ema_model(img_u_w1)
        logits_t_w2, _, _ = ema_model(img_u_w2)
        probs_t_w1 = dist_manager.get_probs(logits_t_w1)
        probs_t_w2 = dist_manager.get_probs(logits_t_w2)
        pseudo_pack = pseudo_labeler(
            teacher_probs_1=probs_t_w1,
            teacher_probs_2=probs_t_w2,
            high_keep_ratio=high_keep,
            mid_keep_ratio=mid_keep,
            min_score_floor=score_floor,
        )
    return pseudo_pack


def set_bn_mode(model, train_mode):
    if train_mode:
        model.apply(lambda m: m.train() if isinstance(m, nn.modules.batchnorm._BatchNorm) else None)
    else:
        model.apply(lambda m: m.eval() if isinstance(m, nn.modules.batchnorm._BatchNorm) else None)


def student_unlabeled_branch(model, dist_manager, img_u_s):
    set_bn_mode(model, False)
    logits_u_s, _, _ = model(img_u_s)
    set_bn_mode(model, True)
    return dist_manager.get_val(logits_u_s)


def unpack_pseudo_maps(pseudo_pack):
    return (
        pseudo_pack["teacher_mu"],
        pseudo_pack["high_weight"],
        pseudo_pack["high_mask"],
        pseudo_pack["mid_weight"],
    )


def midlevel_loss(pred_u_s_phys, teacher_mu, mid_weight, cfg):
    pool_k = max(1, int(cfg["coarse_pool_kernel"]))
    if pool_k > 1:
        pred_u_coarse = F.avg_pool2d(pred_u_s_phys, kernel_size=pool_k, stride=pool_k)
        teacher_u_coarse = F.avg_pool2d(teacher_mu, kernel_size=pool_k, stride=pool_k)
        mid_weight_coarse = F.avg_pool2d(mid_weight, kernel_size=pool_k, stride=pool_k)
        return masked_smooth_l1(pred_u_coarse, teacher_u_coarse, mid_weight_coarse, beta=0.5)
    return masked_smooth_l1(pred_u_s_phys, teacher_mu, mid_weight, beta=0.5)


def rank_piece(pred_u_s_phys, teacher_mu, high_mask, high_weight, dist_manager, cfg, phase_params):
    if phase_params["rank_enabled"]:
        return pairwise_rank_loss_from_maps(
            pred_u_s_phys, teacher_mu, high_mask, high_weight, dist_manager.value_range, cfg
        )
    return pred_u_s_phys.new_tensor(0.0), 0


def combine_unsup_losses(loss_unsup_high, loss_unsup_mid, loss_unsup_rank, loss_sup, phase_params):
    loss_unsup_raw = (
        phase_params["w_unsup_high"] * loss_unsup_high
        + phase_params["w_unsup_mid"] * loss_unsup_mid
        + phase_params["w_unsup_rank"] * loss_unsup_rank
    )
    unsup_cap = phase_params["unsup_cap_ratio"] * loss_sup.detach()
    unsup_scale = torch.clamp(unsup_cap / (loss_unsup_raw.detach() + 1e-8), max=1.0)
    loss_unsup = loss_unsup_raw * unsup_scale
    return loss_unsup_raw, unsup_scale, loss_unsup


def unlabeled_branch(model_pack, img_u, cfg, phase_params):
    img_u_w1, img_u_w2, img_u_s = make_augmented_unlabeled(img_u, phase_params["aug_cfg"])
    pseudo_pack = teacher_pseudo_branch(
        model_pack.ema_model,
        model_pack.dist_manager,
        model_pack.pseudo_labeler,
        img_u_w1,
        img_u_w2,
        phase_params["high_keep"],
        phase_params["mid_keep"],
        phase_params["score_floor"],
    )
    pred_u_s_phys = student_unlabeled_branch(model_pack.model, model_pack.dist_manager, img_u_s)
    teacher_mu, high_weight, high_mask, mid_weight = unpack_pseudo_maps(pseudo_pack)
    loss_unsup_high = masked_smooth_l1(pred_u_s_phys, teacher_mu, high_weight, beta=0.5)
    loss_unsup_mid = midlevel_loss(pred_u_s_phys, teacher_mu, mid_weight, cfg)
    loss_unsup_rank, rank_pairs = rank_piece(
        pred_u_s_phys, teacher_mu, high_mask, high_weight, model_pack.dist_manager, cfg, phase_params
    )
    return {
        "loss_unsup_high": loss_unsup_high,
        "loss_unsup_mid": loss_unsup_mid,
        "loss_unsup_rank": loss_unsup_rank,
        "rank_pairs": rank_pairs,
        "pseudo_pack": pseudo_pack,
    }


def update_meters(meters, sup_pack, unsup_pack, unsup_scale, current_ema_m):
    pseudo_pack = unsup_pack["pseudo_pack"]
    meters["sup_total"] += sup_pack["loss_sup"].item()
    meters["sup_mse"] += sup_pack["loss_mse"].item()
    meters["sup_ssim"] += sup_pack["loss_ssim"].item()
    meters["sup_ldl"] += sup_pack["loss_ldl"].item()
    meters["uns_high"] += unsup_pack["loss_unsup_high"].item()
    meters["uns_mid"] += unsup_pack["loss_unsup_mid"].item()
    meters["uns_rank"] += unsup_pack["loss_unsup_rank"].item()
    meters["uns_scale"] += float(unsup_scale.item())
    meters["rank_pairs"] += float(unsup_pack["rank_pairs"])
    meters["keep_high"] += pseudo_pack["stats"]["keep_high"]
    meters["keep_mid"] += pseudo_pack["stats"]["keep_mid"]
    meters["score_mean"] += pseudo_pack["stats"]["score_mean"]
    meters["std_mean"] += pseudo_pack["stats"]["std_mean"]
    meters["agree_mean"] += pseudo_pack["stats"]["agree_mean"]
    meters["ema_m"] += current_ema_m


def make_postfix(meters, phase_params, denom):
    return {
        "P": phase_params["phase"][:2],
        "Sup": f"{meters['sup_total']/denom:.3f}",
        "Uhi": f"{meters['uns_high']/denom:.3f}",
        "Umid": f"{meters['uns_mid']/denom:.3f}",
        "Urk": f"{meters['uns_rank']/denom:.3f}",
        "Hi": f"{meters['keep_high']/denom:.2f}",
        "Mid": f"{meters['keep_mid']/denom:.2f}",
        "Score": f"{meters['score_mean']/denom:.3f}",
    }


def one_train_turn(step, iter_lb, iter_ulb, cfg, data_pack, total_steps, model_pack, train_state, device, phase_params, meters, pbar):
    img_x, target_phys, img_u = take_batch_pair(iter_lb, iter_ulb, device)
    model_pack.optimizer.zero_grad(set_to_none=True)
    sup_pack = labeled_branch(
        model_pack.model,
        model_pack.dist_manager,
        img_x,
        target_phys,
        cfg,
        data_pack.phys_min,
        data_pack.phys_max,
    )
    unsup_pack = unlabeled_branch(model_pack, img_u, cfg, phase_params)
    _, unsup_scale, loss_unsup = combine_unsup_losses(
        unsup_pack["loss_unsup_high"],
        unsup_pack["loss_unsup_mid"],
        unsup_pack["loss_unsup_rank"],
        sup_pack["loss_sup"],
        phase_params,
    )
    total_loss = sup_pack["loss_sup"] + loss_unsup
    total_loss.backward()
    model_pack.optimizer.step()
    current_ema_m = update_ema_variables(
        model_pack.model,
        model_pack.ema_model,
        train_state.global_step,
        total_steps,
        m_start=phase_params["ema_m_start"],
        m_end=phase_params["ema_m_end"],
    )
    train_state.global_step += 1
    update_meters(meters, sup_pack, unsup_pack, unsup_scale, current_ema_m)
    pbar.set_postfix(make_postfix(meters, phase_params, step + 1))


def phase_desc(epoch, phase_params):
    return f"Ep {epoch} [{phase_params['phase']}|Hi:{phase_params['high_keep']:.2f}|Mid:{phase_params['mid_keep']:.2f}|Floor:{phase_params['score_floor']:.2f}]"


def epoch_average(meters, steps_per_epoch):
    return {
        "avg_score": meters["score_mean"] / steps_per_epoch,
        "avg_std": meters["std_mean"] / steps_per_epoch,
        "avg_agree": meters["agree_mean"] / steps_per_epoch,
        "avg_hi": meters["keep_high"] / steps_per_epoch,
        "avg_mid": meters["keep_mid"] / steps_per_epoch,
    }


def print_epoch_line(epoch, phase_params, val_mse_phys, val_ssim_avg, meters, steps_per_epoch, avg_pack, train_state):
    print(
        f"Ep {epoch:03d} | Phase={phase_params['phase']} | Val MSE: {val_mse_phys:.6f} | Val SSIM: {val_ssim_avg:.4f} | "
        f"Train[Sup={meters['sup_total']/steps_per_epoch:.4f}, Uhi={meters['uns_high']/steps_per_epoch:.4f}, "
        f"Umid={meters['uns_mid']/steps_per_epoch:.4f}, Urank={meters['uns_rank']/steps_per_epoch:.4f}, "
        f"UScale={meters['uns_scale']/steps_per_epoch:.4f}, RankPairs={meters['rank_pairs']/steps_per_epoch:.1f}, "
        f"Hi={avg_pack['avg_hi']:.3f}, Mid={avg_pack['avg_mid']:.3f}, Score={avg_pack['avg_score']:.4f}, Std={avg_pack['avg_std']:.4f}, Agree={avg_pack['avg_agree']:.4f}, "
        f"EMA={meters['ema_m']/steps_per_epoch:.5f}, SinceBest={train_state.anchor_state['epochs_since_best']}]"
    )


def save_best_if_needed(model_pack, student_path, teacher_path, train_state, val_mse_phys):
    improved = val_mse_phys < train_state.best_mse
    if improved:
        train_state.best_mse = val_mse_phys
        train_state.anchor_state["epochs_since_best"] = 0
        torch.save(model_pack.model.state_dict(), student_path)
        torch.save(model_pack.ema_model.state_dict(), teacher_path)
        print(f"New Best Model Saved! (Val Phys MSE: {train_state.best_mse:.6f})")
    else:
        train_state.anchor_state["epochs_since_best"] += 1


def anchor_refine_if_needed(epoch, cfg, model_pack, student_path, teacher_path, train_state, val_mse_phys, avg_pack, device):
    do_switch, switch_reason = should_switch_to_anchor(
        epoch,
        cfg,
        train_state.anchor_state,
        train_state.best_mse,
        val_mse_phys,
        avg_pack["avg_score"],
        avg_pack["avg_std"],
        avg_pack["avg_agree"],
    )
    if do_switch:
        print(f"Switching to anchored refinement at epoch {epoch} | reason={switch_reason}")
        train_state.anchor_state["switched"] = True
        train_state.anchor_state["switch_epoch"] = epoch
        train_state.anchor_state["switch_reason"] = switch_reason
        train_state.anchor_state["epochs_since_best"] = 0
        if bool(cfg["anchor_reload_best"]) and os.path.exists(student_path) and os.path.exists(teacher_path):
            safe_load_state_dict(model_pack.model, student_path, device)
            safe_load_state_dict(model_pack.ema_model, teacher_path, device)
            print(" Reloaded best student/teacher checkpoint before anchored refinement.")
        new_lr = max(float(cfg["lr"]) * float(cfg["anchor_lr_scale"]), 1e-6)
        model_pack.optimizer, model_pack.scheduler = build_optimizer_and_scheduler(
            model_pack.model, new_lr, cfg["epochs"], start_epoch=epoch + 1
        )
        print(f" Rebuilt optimizer/scheduler with anchor lr = {new_lr:.3e}")


def run_one_epoch(epoch, cfg, data_pack, loader_val, steps_per_epoch, total_steps, model_pack, train_state, iter_lb, iter_ulb, student_path, teacher_path, device):
    model_pack.model.train()
    model_pack.ema_model.eval()
    phase_params = get_phase_params(epoch, cfg, train_state.anchor_state)
    pbar = tqdm(range(steps_per_epoch), desc=phase_desc(epoch, phase_params))
    meters = meter_template()
    for step in pbar:
        one_train_turn(step, iter_lb, iter_ulb, cfg, data_pack, total_steps, model_pack, train_state, device, phase_params, meters, pbar)
    model_pack.scheduler.step()
    avg_pack = epoch_average(meters, steps_per_epoch)
    val_mse_phys, val_ssim_avg = evaluate_on_loader(
        model_pack.ema_model,
        loader_val,
        model_pack.dist_manager,
        device,
        data_pack.phys_min,
        data_pack.phys_max,
    )
    print_epoch_line(epoch, phase_params, val_mse_phys, val_ssim_avg, meters, steps_per_epoch, avg_pack, train_state)
    save_best_if_needed(model_pack, student_path, teacher_path, train_state, val_mse_phys)
    anchor_refine_if_needed(epoch, cfg, model_pack, student_path, teacher_path, train_state, val_mse_phys, avg_pack, device)


def final_test_if_ready(cfg, data_pack, loader_test, model_pack, teacher_path, device):
    if loader_test is not None:
        best_teacher = make_wmb(cfg, data_pack.phys_min, data_pack.phys_max, device)
        safe_load_state_dict(best_teacher, teacher_path, device)
        test_mse, test_ssim = evaluate_on_loader(
            best_teacher,
            loader_test,
            model_pack.dist_manager,
            device,
            data_pack.phys_min,
            data_pack.phys_max,
        )
        print(f"[Final Best Teacher] Test Phys MSE: {test_mse:.6f} | Test SSIM: {test_ssim:.4f}")


def main(args):
    cfg = prepare_cfg(args)
    set_seed(cfg["seed"])
    device = get_train_device(cfg)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    print_run_header(cfg)

    data_pack = collect_data_pack(cfg)

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=True)
    if cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2

    bs_lb = safe_batch_size(cfg["bs_lb"], len(data_pack.split.idx_lb))
    bs_ulb = safe_batch_size(cfg["bs_ulb"], len(data_pack.split.idx_ulb))
    bs_eval = safe_batch_size(cfg["bs_eval"], len(data_pack.split.idx_val))

    loader_lb = DataLoader(GPR_Dataset(data_pack.full_data, data_pack.full_labels, data_pack.split.idx_lb, "labeled"), batch_size=bs_lb, shuffle=True, drop_last=True, **loader_kwargs)
    loader_ulb = DataLoader(GPR_Dataset(data_pack.full_data, None, data_pack.split.idx_ulb, "unlabeled"), batch_size=bs_ulb, shuffle=True, drop_last=True, **loader_kwargs)
    loader_val = DataLoader(GPR_Dataset(data_pack.full_data, data_pack.full_labels, data_pack.split.idx_val, "val"), batch_size=bs_eval, shuffle=False, **loader_kwargs)
    loader_test = None
    if data_pack.split.idx_test is not None:
        loader_test = DataLoader(GPR_Dataset(data_pack.full_data, data_pack.full_labels, data_pack.split.idx_test, "test"), batch_size=safe_batch_size(cfg["bs_eval"], len(data_pack.split.idx_test)), shuffle=False, **loader_kwargs)

    steps_per_epoch = max(1, len(loader_lb))
    total_steps = max(1, cfg["epochs"] * steps_per_epoch)

    print(f"[*] Physical range: [{data_pack.phys_min:.6f}, {data_pack.phys_max:.6f}]")
    print(f"[*] Labeled train size: {len(data_pack.split.idx_lb)} | Unlabeled train size: {len(data_pack.split.idx_ulb)}")
    print(f"[*] Batch config -> labeled: {bs_lb}, unlabeled: {bs_ulb}, eval: {bs_eval}")
    print(f"[*] Steps per epoch: {steps_per_epoch}")

    model_pack = prepare_model_pack(cfg, data_pack, device)
    iter_lb = get_infinite_iterator(loader_lb)
    iter_ulb = get_infinite_iterator(loader_ulb)
    train_state = initial_train_state()
    student_path = os.path.join(cfg["save_dir"], cfg["save_student_name"])
    teacher_path = os.path.join(cfg["save_dir"], cfg["save_teacher_name"])

    print("--- Starting staged adaptive-threshold SSL ---")
    for epoch in range(cfg["epochs"]):
        run_one_epoch(
            epoch,
            cfg,
            data_pack,
            loader_val,
            steps_per_epoch,
            total_steps,
            model_pack,
            train_state,
            iter_lb,
            iter_ulb,
            student_path,
            teacher_path,
            device,
        )
    final_test_if_ready(cfg, data_pack, loader_test, model_pack, teacher_path, device)


if __name__ == "__main__":
    main(SimpleNamespace(**CONFIG))
