import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from .model_utils import (
    WindowAttention,
    VanillaAttention,
    STF,
    Mlp,
    window_partition,
    window_reverse,
    get_window_size,
)


class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        configs,
        dim,
        num_heads,
        patch_size=(4, 4, 4),
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        self_attn=False,
        low_res=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = tuple(i // 2 for i in window_size)
        self.self_attn = self_attn

        self.norm1 = norm_layer(dim)
        self.window_attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.shift_window_attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.time_attn = VanillaAttention(
            dim, num_heads, qkv_bias, qk_scale, drop, attn_drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if not low_res:
            # hyper parameter
            self.stf_cell = STF(
                input_shape=(
                    configs["input_length"] // patch_size[0],
                    configs["img_height"] // patch_size[1],
                    configs["img_width"] // patch_size[2],
                ),
                input_dim=dim,
                hidden_dim=dim,
                kernel_size=(3, 3, 3),
            )
        else:
            self.stf_cell = STF(
                input_shape=(
                    configs["input_length"] // patch_size[0],
                    configs["img_height"] // patch_size[1] // 2,
                    configs["img_width"] // patch_size[2] // 2,
                ),
                input_dim=dim,
                hidden_dim=dim,
                kernel_size=(3, 3, 3),
            )

        self.linear_fusion = nn.Linear(2 * dim, dim)

    def attn(
        self,
        window_size,
        x,
        b,
        c,
        dp,
        hp,
        wp,
        memory=None,
        attn_mask=None,
        shift_window=False,
    ):
        if self.self_attn:
            # partition windows
            x_windows = window_partition(x, window_size)  # B*nW, Wd*Wh*Ww, C
            # W-MSA/SW-MSA
            if not shift_window:
                attn_windows = self.window_attn(
                    x_windows, x_windows, x_windows, mask=attn_mask
                )  # B*nW, Wd*Wh*Ww, C
            else:
                attn_windows = self.shift_window_attn(
                    x_windows, x_windows, x_windows, mask=attn_mask
                )  # B*nW, Wd*Wh*Ww, C
            # merge windows
            attn_windows = attn_windows.view(-1, *(window_size + (c,)))
            x = window_reverse(attn_windows, window_size, b, dp, hp, wp)  # B D' H' W' C
        else:
            x_windows = window_partition(x, window_size)  # B*nW, Wd*Wh*Ww, C
            memory_windows = window_partition(memory, window_size)
            # W-MSA/SW-MSA
            if not shift_window:
                attn_windows = self.window_attn(
                    x_windows, memory_windows, memory_windows, mask=attn_mask
                )  # B*nW, Wd*Wh*Ww, C
            else:
                attn_windows = self.shift_window_attn(
                    x_windows, memory_windows, memory_windows, mask=attn_mask
                )  # B*nW, Wd*Wh*Ww, C
            # merge windows
            attn_windows = attn_windows.view(-1, *(window_size + (c,)))
            x = window_reverse(attn_windows, window_size, b, dp, hp, wp)  # B D' H' W' C
        return x

    def forward_part1(self, x, mask_matrix, memory=None):
        memory_tmp = memory  # 暂存memory，为之后time attn做准备
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        if not self.self_attn:
            memory = F.pad(memory, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # window attn
        x = self.attn(
            window_size,
            x,
            B,
            C,
            Dp,
            Hp,
            Wp,
            memory=memory if not self.self_attn else None,
            shift_window=False,
        )

        # shift window attn
        shifted_x = torch.roll(
            x,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(1, 2, 3),
        )
        if not self.self_attn:
            shifted_memory = torch.roll(
                memory,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_memory = None
        x = self.attn(
            window_size,
            shifted_x,
            B,
            C,
            Dp,
            Hp,
            Wp,
            memory=shifted_memory if not self.self_attn else None,
            attn_mask=mask_matrix,
            shift_window=True,
        )
        x = torch.roll(
            x,
            shifts=(shift_size[0], shift_size[1], shift_size[2]),
            dims=(1, 2, 3),
        )

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = rearrange(x, "b t h w c -> (b h w) t c")
        if self.self_attn:
            x = self.time_attn(x, x, x)
        else:
            memory_tmp = rearrange(memory_tmp, "b t h w c -> (b h w) t c")
            x = self.time_attn(x, memory_tmp, memory_tmp)
        x = rearrange(x, "(b h w) t c -> b t h w c", b=B, t=D, h=H, w=W, c=C)

        return x

    def forward_part2(self, x):
        x = self.norm2(x)
        x1 = self.mlp(x)
        x2 = rearrange(x, "b t h w c -> b c t h w")
        x2 = self.stf_cell(x2)
        x2 = rearrange(x2, "b c t h w -> b t h w c")
        x = torch.concat([x1, x2], dim=-1)
        x = self.linear_fusion(x)
        x = self.drop_path(x)
        return x

    def forward(self, x, memory, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, memory=memory, mask_matrix=mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)

        return x
