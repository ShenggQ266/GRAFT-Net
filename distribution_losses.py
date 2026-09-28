import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _to_int_value(x):
    return int(x)
def _to_float_value(x):
    return float(x)
def _lower_bound_float(x, floor):
    return max(float(x), float(floor))
def _range_or_eps(left, right, eps=1e-6):
    return max(float(right) - float(left), eps)
def _same_4d_view(x):
    return x.view(1, -1, 1, 1)
def _softmax_with_temperature(logits, temperature, dim):
    t = _lower_bound_float(temperature, 1e-6)
    return F.softmax(logits / t, dim=dim)
def _weighted_channel_sum(weight, values):
    return (weight * values).sum(dim=1, keepdim=True)

@dataclass
class _DistributionEdges:
    bins: int
    low: float
    high: float
    device: object

    def values(self):
        return _same_4d_view(torch.linspace(self.low, self.high, self.bins, device=self.device))

    def span(self):
        return _range_or_eps(self.low, self.high)


class LinearDistributionManager:
    def __init__(self, num_bins=100, min_val=3.0, max_val=20.0, device='cuda'):
        frame = _DistributionEdges(
            bins=_to_int_value(num_bins),
            low=_to_float_value(min_val),
            high=_to_float_value(max_val),
            device=device,
        )
        self.num_bins = frame.bins
        self.min_val = frame.low
        self.max_val = frame.high
        self.value_range = frame.span()
        self.bin_values = frame.values()
        self.std_max = self.value_range / 2.0

    def get_probs(self, logits, temperature=1.0):
        return _softmax_with_temperature(logits, temperature, dim=1)

    def get_val(self, logits, temperature=1.0):
        probs = self.get_probs(logits, temperature=temperature)
        return _weighted_channel_sum(probs, self.bin_values)

    def get_mean_std_from_probs(self, probs):
        mu = _weighted_channel_sum(probs, self.bin_values)
        diff_sq = (self.bin_values - mu) ** 2
        var = _weighted_channel_sum(probs, diff_sq)
        std = torch.sqrt(var + 1e-6)
        return mu, std

    def generate_gaussian_soft_labels(self, target_phys, sigma=1.0):
        sigma = _lower_bound_float(sigma, 1e-6)
        diff_sq = (self.bin_values - target_phys) ** 2
        return F.softmax(-diff_sq / (2.0 * sigma ** 2), dim=1)


def _gaussian_raw_values(window_size, sigma):
    center = window_size // 2
    sigma_sq = float(2 * sigma ** 2)
    return [math.exp(-(x - center) ** 2 / sigma_sq) for x in range(window_size)]


def _normalize_kernel_1d(values):
    gauss = torch.tensor(values, dtype=torch.float32)
    return gauss / gauss.sum()


def gaussian(window_size, sigma):
    return _normalize_kernel_1d(_gaussian_raw_values(window_size, sigma))


def _outer_2d_from_1d(kernel_1d):
    column = kernel_1d.unsqueeze(1)
    return column.mm(column.t()).float().unsqueeze(0).unsqueeze(0)


def _expand_window_to_channel(window, channel):
    return window.expand(channel, 1, window.size(-2), window.size(-1)).contiguous()


def create_window(window_size, channel):
    _1d = gaussian(window_size, 1.5)
    _2d = _outer_2d_from_1d(_1d)
    return _expand_window_to_channel(_2d, channel)


@dataclass
class _SsimWorkspace:
    img1: object
    img2: object
    window_size: int
    data_range: float

    def channel(self):
        return self.img1.size(1)

    def window(self):
        base = create_window(self.window_size, self.channel())
        return base.to(device=self.img1.device, dtype=self.img1.dtype)

    def conv(self, src, window):
        return F.conv2d(src, window, padding=self.window_size // 2, groups=self.channel())


def _ssim_core_terms(box):
    window = box.window()
    mu1 = box.conv(box.img1, window)
    mu2 = box.conv(box.img2, window)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = box.conv(box.img1 * box.img1, window) - mu1_sq
    sigma2_sq = box.conv(box.img2 * box.img2, window) - mu2_sq
    sigma12 = box.conv(box.img1 * box.img2, window) - mu1_mu2
    return mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12


def _ssim_constants(data_range):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return c1, c2


def _ssim_map_from_terms(terms, constants):
    mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = terms
    c1, c2 = constants
    top = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    bottom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    return top / bottom


def ssim_score(img1, img2, window_size=11, size_average=True, data_range=1.0):
    box = _SsimWorkspace(img1=img1, img2=img2, window_size=window_size, data_range=data_range)
    ssim_map = _ssim_map_from_terms(_ssim_core_terms(box), _ssim_constants(data_range))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def _masked_reduce(loss_map, mask, eps):
    return (loss_map * mask).sum() / (mask.sum() + eps)


def masked_smooth_l1(pred, target, mask, beta=1.0, eps=1e-6):
    loss_map = F.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    return _masked_reduce(loss_map, mask, eps)


def masked_mse(pred, target, mask, eps=1e-6):
    loss_map = (pred - target) ** 2
    return _masked_reduce(loss_map, mask, eps)


def masked_mean(x, mask, eps=1e-6):
    return _masked_reduce(x, mask, eps)


@dataclass
class _RankLossCfg:
    pool_k: int
    min_valid_ratio: float
    pairs_per_image: int
    teacher_margin: float
    temperature: float
    weight_power: float

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            pool_k=max(1, int(cfg["rank_pool_kernel"])),
            min_valid_ratio=float(cfg["rank_min_valid_ratio"]),
            pairs_per_image=max(1, int(cfg["rank_pairs_per_image"])),
            teacher_margin=float(cfg["rank_teacher_margin"]),
            temperature=max(float(cfg["rank_temperature"]), 1e-6),
            weight_power=float(cfg["rank_weight_power"]),
        )


def _maybe_pool_rank_maps(student_map, teacher_map, valid_mask, reliability_weight, pool_k):
    if pool_k > 1:
        s_map = F.avg_pool2d(student_map, kernel_size=pool_k, stride=pool_k)
        t_map = F.avg_pool2d(teacher_map, kernel_size=pool_k, stride=pool_k)
        v_map = F.avg_pool2d(valid_mask, kernel_size=pool_k, stride=pool_k)
        r_map = F.avg_pool2d(reliability_weight, kernel_size=pool_k, stride=pool_k)
        return s_map, t_map, v_map, r_map
    return student_map, teacher_map, valid_mask, reliability_weight


def _valid_grid_from_ratio(v_map, min_valid_ratio):
    return v_map >= min_valid_ratio


def _rand_pair_indices(n_valid, sample_pairs, device):
    idx1 = torch.randint(0, n_valid, (sample_pairs,), device=device)
    idx2 = torch.randint(0, n_valid, (sample_pairs,), device=device)
    diff_pair = idx1 != idx2
    return idx1, idx2, diff_pair


def _coords_from_pair_index(coords, idx1, idx2, diff_pair):
    if not diff_pair.any():
        return None, None
    return coords[idx1[diff_pair]], coords[idx2[diff_pair]]


def _two_map_pick(map2d, b, coord1, coord2):
    v1 = map2d[b, 0, coord1[:, 0], coord1[:, 1]]
    v2 = map2d[b, 0, coord2[:, 0], coord2[:, 1]]
    return v1, v2


def _strong_teacher_pairs(tdiff, teacher_margin):
    return tdiff.abs() >= teacher_margin


def _sign_from_teacher_diff(tdiff):
    target_sign = torch.sign(tdiff)
    return torch.where(target_sign == 0.0, torch.ones_like(target_sign), target_sign)


def _pair_weight_from_maps(r_map, b, coord1, coord2, tdiff, value_range, weight_power):
    r1, r2 = _two_map_pick(r_map, b, coord1, coord2)
    pair_reliability = (r1 * r2).clamp(min=0.0, max=1.0)
    pair_strength = (tdiff.abs() / max(float(value_range), 1e-6)).clamp(0.0, 1.0).pow(weight_power)
    return (pair_reliability * pair_strength).detach()


def _rank_loss_piece(sdiff, target_sign, pair_weight, temperature):
    rank_loss = F.softplus(-(target_sign * sdiff) / temperature)
    return (rank_loss * pair_weight).sum() / (pair_weight.sum() + 1e-6)


def _collect_rank_for_one_image(b, s_map, t_map, valid, r_map, value_range, rcfg):
    coords = valid[b, 0].nonzero(as_tuple=False)
    n_valid = int(coords.size(0))
    if n_valid < 2:
        return None, 0

    idx1, idx2, diff_pair = _rand_pair_indices(n_valid, rcfg.pairs_per_image, s_map.device)
    coord1, coord2 = _coords_from_pair_index(coords, idx1, idx2, diff_pair)
    if coord1 is None:
        return None, 0

    t1, t2 = _two_map_pick(t_map, b, coord1, coord2)
    tdiff = t1 - t2

    strong_pair = _strong_teacher_pairs(tdiff, rcfg.teacher_margin)
    if not strong_pair.any():
        return None, 0

    coord1 = coord1[strong_pair]
    coord2 = coord2[strong_pair]
    tdiff = tdiff[strong_pair]
    target_sign = _sign_from_teacher_diff(tdiff)

    s1, s2 = _two_map_pick(s_map, b, coord1, coord2)
    sdiff = s1 - s2
    pair_weight = _pair_weight_from_maps(r_map, b, coord1, coord2, tdiff, value_range, rcfg.weight_power)
    rank_loss = _rank_loss_piece(sdiff, target_sign, pair_weight, rcfg.temperature)
    return rank_loss, int(tdiff.numel())


def pairwise_rank_loss_from_maps(student_map, teacher_map, valid_mask, reliability_weight, value_range, cfg):
    rcfg = _RankLossCfg.from_cfg(cfg)
    s_map, t_map, v_map, r_map = _maybe_pool_rank_maps(
        student_map, teacher_map, valid_mask, reliability_weight, rcfg.pool_k
    )
    valid = _valid_grid_from_ratio(v_map, rcfg.min_valid_ratio)
    pair_losses = []
    pair_counter = 0

    for b in range(s_map.size(0)):
        rank_loss, used_pairs = _collect_rank_for_one_image(
            b, s_map, t_map, valid, r_map, value_range, rcfg
        )
        if rank_loss is None:
            continue
        pair_losses.append(rank_loss)
        pair_counter += used_pairs

    if not pair_losses:
        return student_map.new_tensor(0.0), 0

    return torch.stack(pair_losses).mean(), pair_counter
