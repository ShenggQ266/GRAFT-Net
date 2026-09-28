import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    print("Warning.")
    DWTForward, DWTInverse = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning.")


def _as_contiguous_channels_first(x):
    return x.permute(0, 3, 1, 2).contiguous()


def _as_contiguous_channels_last(x):
    return x.permute(0, 2, 3, 1).contiguous()


def _needs_resize_2d(x, target_size):
    if target_size is None:
        return False
    if len(target_size) != 2:
        return False
    if int(target_size[0]) <= 0 or int(target_size[1]) <= 0:
        return False
    return x.shape[2:] != tuple(target_size)


def _resize_if_needed(x, target_size):
    if _needs_resize_2d(x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    return x


def _take_drop_path(drop_path, i):
    if isinstance(drop_path, list):
        return drop_path[i]
    return drop_path


class _DprCursor:
    def __init__(self, values):
        self.values = values
        self.pos = 0

    def take(self, n):
        out = self.values[self.pos:self.pos + n]
        self.pos += n
        return out


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._assign_sizes(d_model, d_state, d_conv, expand, dt_rank)
        self._build_entry_layers(factory_kwargs, conv_bias, bias)
        self._build_scan_projection_weights(factory_kwargs)
        self._prepare_dt_layers(factory_kwargs, dt_min, dt_max, dt_scale, dt_init_floor)
        self._build_state_parameters(factory_kwargs, bias, dropout)

    def _assign_sizes(self, d_model, d_state, d_conv, expand, dt_rank):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

    def _build_entry_layers(self, factory_kwargs, conv_bias, bias):
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=self.d_conv, padding=(self.d_conv - 1) // 2,
                                **factory_kwargs)
        self.act = nn.SiLU()

    def _build_scan_projection_weights(self, factory_kwargs):
        self.x_proj_weight = nn.Parameter(
            torch.empty(4, (self.dt_rank + self.d_state * 2), self.d_inner, **factory_kwargs))
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank, **factory_kwargs))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner, **factory_kwargs))

    def _prepare_dt_layers(self, factory_kwargs, dt_min, dt_max, dt_scale, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_projs_weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(4, self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt)

    def _build_state_parameters(self, factory_kwargs, bias, dropout):
        self.A_logs = nn.Parameter(
            torch.log(repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> (r d) n", r=4,
                             d=self.d_inner)))
        self.Ds = nn.Parameter(torch.ones(4, self.d_inner))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def _scan_views(self, x, b, l):
        h_view = x.view(b, -1, l)
        w_view = torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, -1, l)
        x_hwwh = torch.stack([h_view, w_view], dim=1).view(b, 2, -1, l)
        return torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

    def _project_scan_views(self, xs, b, k, l):
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)
        return dts, bs, cs

    def _flat_scan_inputs(self, xs, dts, bs, cs, b, k, l):
        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        bs = bs.float().view(b, k, -1, l)
        cs = cs.float().view(b, k, -1, l)
        ds = self.Ds.float().view(-1)
        a_logs = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        return xs, dts, bs, cs, ds, a_logs, dt_bias

    def _scan_or_passthrough(self, xs, dts, a_logs, bs, cs, ds, dt_bias, b, k, l):
        if selective_scan_fn is None:
            return xs.view(b, k, -1, l)
        return selective_scan_fn(xs, dts, a_logs, bs, cs, ds, z=None, delta_bias=dt_bias, delta_softplus=True,
                                 return_last_state=False).view(b, k, -1, l)

    def _return_scan_directions(self, out_y, b, h, w, l):
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, l)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3).contiguous().view(b, -1, l)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3).contiguous().view(b, -1, l)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_core(self, x: torch.Tensor):
        b, c, h, w = x.shape
        l = h * w
        k = 4
        xs = self._scan_views(x, b, l)
        dts, bs, cs = self._project_scan_views(xs, b, k, l)
        xs, dts, bs, cs, ds, a_logs, dt_bias = self._flat_scan_inputs(xs, dts, bs, cs, b, k, l)
        out_y = self._scan_or_passthrough(xs, dts, a_logs, bs, cs, ds, dt_bias, b, k, l)
        return self._return_scan_directions(out_y, b, h, w, l)

    def _in_projection_path(self, x):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = _as_contiguous_channels_first(x)
        x = self.act(self.conv2d(x))
        return x, z

    def _core_sum(self, x):
        y1, y2, y3, y4 = self.forward_core(x)
        return y1 + y2 + y3 + y4

    def _output_projection_path(self, y, z, b, h, w):
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return self.dropout(out)

    def forward(self, x: torch.Tensor, **kwargs):
        b, h, w, c = x.shape
        x, z = self._in_projection_path(x)
        y = self._core_sum(x)
        return self._output_projection_path(y, z, b, h, w)


class WaveletFreqBranch(nn.Module):
    def __init__(self, dim, wave='db2'):
        super().__init__()
        if DWTForward is not None:
            self.dwt = DWTForward(J=1, mode='zero', wave=wave)
            self.idwt = DWTInverse(mode='zero', wave=wave)
        self.low_freq_net = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), nn.BatchNorm2d(dim), nn.GELU(),
                                          nn.Conv2d(dim, dim, 1))
        self.high_freq_net = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim), nn.BatchNorm2d(dim * 3),
                                           nn.GELU())
        self.freq_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim * 3, dim // 4, 1), nn.ReLU(),
                                       nn.Conv2d(dim // 4, dim * 3, 1), nn.Sigmoid())
        self.gain_scale = nn.Parameter(torch.ones(1, dim * 3, 1, 1))

    def _split_image_layout(self, x):
        b, h, w, c = x.shape
        return b, h, w, c, _as_contiguous_channels_first(x)

    def _pad_image(self, x_img, h, w):
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x_img = F.pad(x_img, (0, pad_w, 0, pad_h), mode='reflect')
        return x_img, pad_h, pad_w

    def _frequency_pass(self, x_img, b, c):
        yl, yh = self.dwt(x_img)
        yl_out = self.low_freq_net(yl)
        yh_flat = yh[0].view(b, -1, yl.shape[2], yl.shape[3])
        feat_h = self.high_freq_net(yh_flat)
        yh_enhanced = feat_h * self.freq_gate(feat_h) * self.gain_scale
        rec_pack = (yl_out, [yh_enhanced.view(b, c, 3, yl.shape[2], yl.shape[3])])
        return self.idwt(rec_pack)

    def _crop_back(self, x_recon, h, w, pad_h, pad_w):
        if pad_h or pad_w:
            x_recon = x_recon[:, :, :h, :w]
        return x_recon

    def forward(self, x):
        if DWTForward is None:
            return x
        b, h, w, c, x_img = self._split_image_layout(x)
        x_img, pad_h, pad_w = self._pad_image(x_img, h, w)
        x_recon = self._frequency_pass(x_img, b, c)
        x_recon = self._crop_back(x_recon, h, w, pad_h, pad_w)
        return _as_contiguous_channels_last(x_recon)


class WaveletMambaBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., d_state=16):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.spatial_path = SS2D(d_model=hidden_dim, d_state=d_state)
        self.freq_path = WaveletFreqBranch(dim=hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _paired_paths(self, x_norm):
        return self.spatial_path(x_norm), self.freq_path(x_norm)

    def _fuse_pair(self, left, right):
        return self.fusion(torch.cat([left, right], dim=-1))

    def forward(self, input):
        x_norm = self.ln_1(input)
        x_a, x_b = self._paired_paths(x_norm)
        x_fused = self._fuse_pair(x_a, x_b)
        return input + self.drop_path(x_fused)


class WaveletMambaLayer(nn.Module):
    def __init__(self, dim, depth, drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList(
            [WaveletMambaBlock(dim, _take_drop_path(drop_path, i)) for i in range(depth)])

    def _apply_one(self, x, block):
        return block(x)

    def forward(self, x):
        for blk in self.blocks:
            x = self._apply_one(x, blk)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def _pad_corner(self, x, h, w):
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        return x

    def _collect_quadrants(self, x):
        return torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)

    def forward(self, x):
        b, h, w, c = x.shape
        x = self._pad_corner(x, h, w)
        x = self._collect_quadrants(x)
        return self.reduction(self.norm(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU(True),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.chunk_dim = in_channels // 2
        self.conv_local = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, dilation=1, groups=self.chunk_dim)
        self.conv_context = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 3, dilation=3, groups=self.chunk_dim)
        self.fuse = nn.Conv2d(in_channels, out_channels, 1)

    def _split_local_context(self, x):
        return torch.split(x, self.chunk_dim, dim=1)

    def forward(self, x):
        x1, x2 = self._split_local_context(x)
        return self.fuse(torch.cat([self.conv_local(x1), self.conv_context(x2)], dim=1))


class MDA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // factor, channels // factor)
        self.conv1x1 = nn.Conv2d(channels // factor, channels // factor, 1)
        self.conv_ms = MultiScaleDilatedConv(channels // factor, channels // factor)

    def _group_input(self, x):
        b, c, h, w = x.size()
        return b, c, h, w, x.reshape(b * self.groups, -1, h, w)

    def _axis_gate(self, group_x, h, w):
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        return torch.split(hw, [h, w], dim=2)

    def _gated_norm(self, group_x, x_h, x_w):
        return self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

    def _soft_axis(self, x):
        return F.softmax(self.agp(x).reshape(x.size(0), -1, 1).permute(0, 2, 1), dim=-1)

    def _make_weight_map(self, x1, x2, b, c, h, w):
        x11 = self._soft_axis(x1)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self._soft_axis(x2)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        return (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

    def forward(self, x):
        b, c, h, w, group_x = self._group_input(x)
        x_h, x_w = self._axis_gate(group_x, h, w)
        x1 = self._gated_norm(group_x, x_h, x_w)
        x2 = self.conv_ms(group_x)
        weights = self._make_weight_map(x1, x2, b, c, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class Up_MDA(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        mid_channels = in_channels // 2 + skip_channels
        self.conv_align = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 1), nn.BatchNorm2d(mid_channels),
                                        nn.ReLU())
        self.md_ema = MDA(mid_channels, factor=8)
        self.conv_out = DoubleConv(mid_channels, out_channels)

    def _up_then_match(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return x1

    def _skip_merge_refine(self, x1, x2):
        return self.md_ema(self.conv_align(torch.cat([x2, x1], dim=1)))

    def forward(self, x1, x2):
        x1 = self._up_then_match(x1, x2)
        x_refined = self._skip_merge_refine(x1, x2)
        return self.conv_out(x_refined)


class WaveletMambaBinningUNet(nn.Module):
    def __init__(self, n_channels=1, num_bins=0, min_val=0, max_val=0, depths=[2, 2, 9, 2],
                 target_size=(0, 0)):
        super().__init__()
        dims = [64, 128, 256, 512, 1024]
        self.target_size = target_size
        self.target_size = target_size
        self.num_bins = num_bins
        self._register_bins(min_val, max_val, num_bins)
        self._build_entry(n_channels, dims)
        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]
        cursor = _DprCursor(dpr)
        self._build_down_stack(dims, depths, cursor)
        self._build_up_stack(dims)
        self.outc = nn.Conv2d(dims[0], num_bins, 1)

    def _register_bins(self, min_val, max_val, num_bins):
        bin_step = (max_val - min_val) / num_bins
        centers = torch.linspace(min_val + bin_step / 2, max_val - bin_step / 2, num_bins)
        self.register_buffer('bin_centers', centers)

    def _build_entry(self, n_channels, dims):
        self.inc = DoubleConv(n_channels, dims[0])

    def _build_down_stack(self, dims, depths, cursor):
        self.down1 = PatchMerging2D(dim=dims[0])
        self.layer1 = WaveletMambaLayer(dim=dims[1], depth=depths[0], drop_path=cursor.take(depths[0]))
        self.down2 = PatchMerging2D(dim=dims[1])
        self.layer2 = WaveletMambaLayer(dim=dims[2], depth=depths[1], drop_path=cursor.take(depths[1]))
        self.down3 = PatchMerging2D(dim=dims[2])
        self.layer3 = WaveletMambaLayer(dim=dims[3], depth=depths[2], drop_path=cursor.take(depths[2]))
        self.down4 = PatchMerging2D(dim=dims[3])
        self.layer4 = WaveletMambaLayer(dim=dims[4], depth=depths[3], drop_path=cursor.take(depths[3]))

    def _build_up_stack(self, dims):
        self.up1 = Up_MDA(dims[4], dims[3], dims[3])
        self.up2 = Up_MDA(dims[3], dims[2], dims[2])
        self.up3 = Up_MDA(dims[2], dims[1], dims[1])
        self.up4 = Up_MDA(dims[1], dims[0], dims[0])

    def _encode(self, x):
        s1 = self.inc(x)
        x_down = self.layer1(self.down1(s1.permute(0, 2, 3, 1)))
        s2 = x_down.permute(0, 3, 1, 2)
        x_down = self.layer2(self.down2(x_down))
        s3 = x_down.permute(0, 3, 1, 2)
        x_down = self.layer3(self.down3(x_down))
        s4 = x_down.permute(0, 3, 1, 2)
        x_down = self.layer4(self.down4(x_down))
        bottleneck = x_down.permute(0, 3, 1, 2)
        return s1, s2, s3, s4, bottleneck

    def _decode(self, s1, s2, s3, s4, bottleneck):
        d4 = self.up1(bottleneck, s4)
        d3 = self.up2(d4, s3)
        d2 = self.up3(d3, s2)
        return self.up4(d2, s1)

    def _head(self, feature):
        return _resize_if_needed(self.outc(feature), self.target_size)

    def _head_fp(self, feature):
        feature_fp = F.dropout2d(feature, p=0.2, training=True)
        return _resize_if_needed(self.outc(feature_fp), self.target_size)

    def _value_from_logits(self, out_logits):
        probs = F.softmax(out_logits, dim=1)
        return torch.sum(probs * self.bin_centers.view(1, -1, 1, 1), dim=1, keepdim=True)

    def forward(self, x, need_fp=False):
        s1, s2, s3, s4, bottleneck = self._encode(x)
        feature = self._decode(s1, s2, s3, s4, bottleneck)
        if need_fp:
            return self._head_fp(feature)
        out_logits = self._head(feature)
        out_val = self._value_from_logits(out_logits)
        return out_logits, out_val, feature
