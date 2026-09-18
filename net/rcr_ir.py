from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcr_ir_utils import haar_dwt2d


@dataclass
class RCRIRConfig:
    in_ch: int = 3
    out_ch: int = 3
    base_dim: int = 64
    dim_multipliers: Tuple[int, int, int, int] = (1, 2, 3, 4)
    enc_blocks: Tuple[int, int, int, int] = (2, 2, 4, 2)
    dec_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2)
    z_dim: int = 32
    assb_blocks: int = 2
    attention_heads: int = 4
    ffn_expansion: float = 2.0
    layernorm_type: str = "WithBias"
    bias: bool = False


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, n, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


def _make_divisible(v: int, divisor: int) -> int:
    return max(divisor, int(math.ceil(v / float(divisor))) * divisor)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, layernorm_type: str = "WithBias"):
        super().__init__()
        self.body = (BiasFreeLayerNorm(dim) if layernorm_type == "BiasFree"
                     else WithBiasLayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0, bias: bool = False):
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SpatialGate(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias)
        self.proj = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(self.dwconv(x)))


class ERAB(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        bias: bool = False,
        layernorm_type: str = "WithBias",
    ):
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        self.norm1 = LayerNorm2d(dim, layernorm_type)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1,
            groups=hidden, bias=bias)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.channel_gate = SqueezeExcitation(dim)
        self.spatial_gate = SpatialGate(dim, bias=bias)
        self.norm2 = LayerNorm2d(dim, layernorm_type)
        self.ffn = FeedForward(dim, expansion=expansion, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw2(F.gelu(self.dwconv(self.pw1(y))))
        y = y * self.channel_gate(y) * self.spatial_gate(y)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class ReliabilityEstimator(nn.Module):
    def __init__(self, dim: int, z_dim: int, bias: bool = False):
        super().__init__()
        state_hidden = max(dim, z_dim * 2)
        self.low_filter = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.reliability_proj = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(dim * 2, state_hidden),
            nn.GELU(),
            nn.Linear(state_hidden, z_dim),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh = haar_dwt2d(feat)
        low = self.low_filter(ll)
        reliability = torch.sigmoid(self.reliability_proj(low))
        reliability = F.interpolate(
            reliability, size=feat.shape[-2:],
            mode="bilinear", align_corners=False)

        pooled_low = ll.mean(dim=(-2, -1))
        pooled_high = (lh.abs() + hl.abs() + hh.abs()).mean(dim=(-2, -1))
        state = self.state_mlp(torch.cat([pooled_low, pooled_high], dim=1))
        return reliability, state


class ConflictRestrictedResidualAdapter(nn.Module):
    def __init__(self, dim: int, z_dim: int, bias: bool = False):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.channel_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.channel_dw = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1,
            groups=hidden, bias=bias)
        self.channel_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.spatial_dw = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias)
        self.spatial_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        gate_hidden = max(z_dim * 2, dim // 2)
        self.alpha = nn.Sequential(
            nn.Linear(z_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(z_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, dim),
        )

    def forward(self, feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        channel_branch = self.channel_out(
            F.gelu(self.channel_dw(self.channel_in(feat))))
        spatial_branch = self.spatial_out(self.spatial_dw(feat))

        alpha = torch.sigmoid(self.alpha(state)).unsqueeze(-1).unsqueeze(-1)
        beta = torch.sigmoid(self.beta(state)).unsqueeze(-1).unsqueeze(-1)
        correction = alpha * channel_branch + beta * spatial_branch
        return correction


class EncoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        z_dim: int,
        expansion: float = 2.0,
        bias: bool = False,
        layernorm_type: str = "WithBias",
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ERAB(dim, expansion=expansion, bias=bias,
                 layernorm_type=layernorm_type)
            for _ in range(num_blocks)
        ])
        self.rme = ReliabilityEstimator(dim, z_dim, bias=bias)
        self.crra = ConflictRestrictedResidualAdapter(dim, z_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.blocks(x)
        reliability, state = self.rme(feat)
        shared = reliability * feat
        conflict = (1.0 - reliability) * feat
        correction = self.crra(conflict, state)
        skip = shared + correction
        return feat, skip, reliability


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.body = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0,
                          mode="bilinear", align_corners=False)
        return self.proj(x)


class DirectionalStateSpaceMix(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.gate_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.local = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _scan(value: torch.Tensor,
              gate: torch.Tensor,
              dim: int) -> torch.Tensor:
        forward_num = torch.cumsum(value * gate, dim=dim)
        forward_den = torch.cumsum(gate, dim=dim).clamp_min(1e-4)
        backward_num = torch.flip(
            torch.cumsum(torch.flip(value * gate, dims=[dim]), dim=dim),
            dims=[dim])
        backward_den = torch.flip(
            torch.cumsum(torch.flip(gate, dims=[dim]), dim=dim),
            dims=[dim]).clamp_min(1e-4)
        return 0.5 * (forward_num / forward_den + backward_num / backward_den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.local(self.value_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))
        horiz = self._scan(value, gate, dim=-1)
        vert = self._scan(value, gate, dim=-2)
        return self.out_proj(0.5 * (horiz + vert))


class LiteAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, bias: bool = False):
        super().__init__()
        self.heads = max(int(heads), 1)
        inner = _make_divisible(max(dim // 2, self.heads), self.heads)
        self.inner = inner
        self.head_dim = inner // self.heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, inner, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, inner, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, inner, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(inner, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n = h * w

        q = self.q_proj(x).reshape(b, self.heads, self.head_dim, n)
        k = self.k_proj(x).reshape(b, self.heads, self.head_dim, n)
        v = self.v_proj(x).reshape(b, self.heads, self.head_dim, n)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.inner, h, w)
        return self.out_proj(out)


class ASSBLite(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        expansion: float = 2.0,
        bias: bool = False,
        layernorm_type: str = "WithBias",
    ):
        super().__init__()
        self.local = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.norm1 = LayerNorm2d(dim, layernorm_type)
        self.ssm = DirectionalStateSpaceMix(dim, bias=bias)
        self.norm2 = LayerNorm2d(dim, layernorm_type)
        self.attn = LiteAttention(dim, heads=heads, bias=bias)
        self.norm3 = LayerNorm2d(dim, layernorm_type)
        self.ffn = FeedForward(dim, expansion=expansion, bias=bias)
        self.recalibrate = SqueezeExcitation(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x)
        x = x + self.ssm(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x * self.recalibrate(x) + x

class DecoderStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        num_blocks: int,
        expansion: float = 2.0,
        bias: bool = False,
        layernorm_type: str = "WithBias",
    ):
        super().__init__()
        self.merge = nn.Conv2d(
            in_dim + skip_dim, out_dim, kernel_size=1, bias=bias)
        self.blocks = nn.Sequential(*[
            ERAB(out_dim, expansion=expansion, bias=bias,
                 layernorm_type=layernorm_type)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.merge(torch.cat([x, skip], dim=1))
        feat = self.blocks(feat)
        return feat


class RCRIR(nn.Module):
    def __init__(self, cfg: RCRIRConfig | None = None):
        super().__init__()
        self.cfg = cfg or RCRIRConfig()
        cfg = self.cfg

        if len(cfg.dim_multipliers) != 4:
            raise ValueError("dim_multipliers must have length 4")
        if len(cfg.enc_blocks) != 4 or len(cfg.dec_blocks) != 4:
            raise ValueError("enc_blocks and dec_blocks must both have length 4")

        dims = [cfg.base_dim * m for m in cfg.dim_multipliers]
        bias = bool(cfg.bias)

        self.stem = nn.Conv2d(cfg.in_ch, dims[0], kernel_size=3, padding=1, bias=bias)

        self.enc1 = EncoderStage(
            dims[0], cfg.enc_blocks[0], cfg.z_dim,
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.enc2 = EncoderStage(
            dims[1], cfg.enc_blocks[1], cfg.z_dim,
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.enc3 = EncoderStage(
            dims[2], cfg.enc_blocks[2], cfg.z_dim,
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.enc4 = EncoderStage(
            dims[3], cfg.enc_blocks[3], cfg.z_dim,
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)

        self.down1 = Downsample(dims[0], dims[1], bias=bias)
        self.down2 = Downsample(dims[1], dims[2], bias=bias)
        self.down3 = Downsample(dims[2], dims[3], bias=bias)

        self.bottleneck = nn.Sequential(*[
            ASSBLite(
                dims[3], heads=cfg.attention_heads,
                expansion=cfg.ffn_expansion, bias=bias,
                layernorm_type=cfg.layernorm_type)
            for _ in range(cfg.assb_blocks)
        ])

        self.dec4 = DecoderStage(
            dims[3], dims[3], dims[3], cfg.dec_blocks[0],
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.up3 = Upsample(dims[3], dims[2], bias=bias)
        self.dec3 = DecoderStage(
            dims[2], dims[2], dims[2], cfg.dec_blocks[1],
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.up2 = Upsample(dims[2], dims[1], bias=bias)
        self.dec2 = DecoderStage(
            dims[1], dims[1], dims[1], cfg.dec_blocks[2],
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)
        self.up1 = Upsample(dims[1], dims[0], bias=bias)
        self.dec1 = DecoderStage(
            dims[0], dims[0], dims[0], cfg.dec_blocks[3],
            expansion=cfg.ffn_expansion, bias=bias,
            layernorm_type=cfg.layernorm_type)

        self.final_refine = nn.Sequential(
            ERAB(dims[0], expansion=cfg.ffn_expansion, bias=bias,
                 layernorm_type=cfg.layernorm_type),
            ERAB(dims[0], expansion=cfg.ffn_expansion, bias=bias,
                 layernorm_type=cfg.layernorm_type),
        )
        self.final_out = nn.Conv2d(
            dims[0], cfg.out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, return_reliability: bool = False):
        x0 = self.stem(x)

        f1, s1, r1 = self.enc1(x0)
        f2, s2, r2 = self.enc2(self.down1(f1))
        f3, s3, r3 = self.enc3(self.down2(f2))
        f4, s4, r4 = self.enc4(self.down3(f3))

        bottleneck = self.bottleneck(f4)

        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(self.up3(d4), s3)
        d2 = self.dec2(self.up2(d3), s2)
        d1 = self.dec1(self.up1(d2), s1)

        out = x + self.final_out(self.final_refine(d1))
        if return_reliability:
            return out, [r1, r2, r3, r4]
        return out

