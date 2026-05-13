import copy
import math
from functools import reduce, lru_cache
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class STF(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size):
        super(STF, self).__init__()
        self.kernel_size = kernel_size
        padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2
        self.module = nn.Sequential(
            nn.Conv3d(
                input_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=(1, 1, 1),
                padding=padding,
            ),
            nn.LayerNorm([hidden_dim, input_shape[0], input_shape[1], input_shape[2]]),
            nn.SiLU(),
            nn.Conv3d(
                hidden_dim,
                input_dim,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            ),
            nn.LayerNorm([hidden_dim, input_shape[0], input_shape[1], input_shape[2]]),
        )

    def forward(self, x):
        return self.module(x)


class VanillaAttention(nn.Module):
    def __init__(
        self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_linear = clones(nn.Linear(dim, dim, bias=qkv_bias), 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        B_, N, C = q.shape
        q, k, v = [
            _l(x).view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            for _l, x in zip(self.qkv_linear, (q, k, v))
        ]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_linear = clones(nn.Linear(dim, dim, bias=qkv_bias), 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        B_, N, C = q.shape
        q, k, v = [
            _l(x).view(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            for _l, x in zip(self.qkv_linear, (q, k, v))
        ]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(
            N, N, -1
        )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, dim * 2)
        self.up_sample = nn.PixelShuffle(2)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, _, _, _, _ = x.shape
        x = self.expand(x)
        x = rearrange(x, "b d h w c -> (b d) c h w")
        x = self.up_sample(x)
        x = rearrange(x, "(b d) c h w -> b d h w c", b=B)
        x = self.norm(x)
        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class MultiScalePatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size_t,
        patch_size_h,
        patch_size_w,
        in_chans=1,
        embed_dim=128,
        norm_layer=None,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ModuleList()
        for i in range(len(patch_size_h)):
            ps = (patch_size_t[i], patch_size_h[i], patch_size_w[i])
            if i == len(patch_size_h) - 1:
                dim = embed_dim // 2**i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = (patch_size_t[0], patch_size_h[0], patch_size_w[0])
            padding = (
                math.ceil((ps[0] - patch_size_t[0]) / 2),
                math.ceil((ps[1] - patch_size_h[0]) / 2),
                math.ceil((ps[2] - patch_size_w[0]) / 2),
            )
            self.proj.append(
                nn.Conv3d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding)
            )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        xs = []
        for i in range(len(self.proj)):
            tx = self.proj[i](x)
            xs.append(tx)  # b c t ph pw
        x = torch.cat(xs, dim=1)
        if self.norm is not None:
            c, ph, pw = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, c, ph, pw)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)  # b c t ph pw
        if self.norm is not None:
            c, ph, pw = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, c, ph, pw)
        return x


class PatchEmbedBack3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose3d(
            embed_dim, in_chans, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # B C D Wh Ww
        return x


class ConvOut(nn.Module):
    def __init__(self, configs, in_channel, out_channel):
        super().__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.LayerNorm(
                [in_channel // 2, configs["img_height"], configs["img_width"]]
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channel // 2, in_channel // 4, kernel_size=3, padding=1, bias=False
            ),
            nn.LayerNorm(
                [in_channel // 4, configs["img_height"], configs["img_width"]]
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channel // 4, out_channel, kernel_size=3, padding=1, bias=False
            ),
            nn.LayerNorm([out_channel, configs["img_height"], configs["img_width"]]),
        )

    def forward(self, x):
        B, _, _, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv_out(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x
