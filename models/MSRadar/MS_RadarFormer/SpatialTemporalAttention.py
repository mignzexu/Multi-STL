import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .SpatialTemporalBlock import SpatialTemporalAttentionBlock
from .model_utils import (
    MultiScalePatchEmbed3D,
    get_window_size,
    compute_mask,
    PatchMerging,
    PatchEmbed3D,
)


class BasicLayer(nn.Module):
    def __init__(
        self,
        configs,
        dim,
        depth,
        num_heads,
        patch_size=(4, 4, 4),
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=None,
        norm_layer=nn.LayerNorm,
        self_attn=False,
        low_res=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.self_attn = self_attn
        self.depth = depth

        self.shift_size = tuple(i // 2 for i in window_size)
        if drop_path is None:
            drop_path = [0.0 for _ in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(
                    configs,
                    patch_size=patch_size,
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    self_attn=self_attn,
                    low_res=low_res,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, memory=None):
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )
        x = rearrange(x, "b c d h w -> b d h w c")
        if not self.self_attn:
            memory = rearrange(memory, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        if not self.self_attn:
            for blk in self.blocks:
                x = blk(x, memory, attn_mask)
        else:
            for blk in self.blocks:
                x = blk(x, None, attn_mask)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class SpatialTemporalAttention(nn.Module):
    def __init__(
        self,
        configs,
        patch_size,
        embed_dim,
        in_chans,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        self_attn=False,
        require_patch_embed=False,
        down_sample=False,
        low_res=False,
    ):
        """
        :param patch_size:
        :param embed_dim:
        :param depths: 模块个数
        :param num_heads:
        :param window_size:
        :param mlp_ratio:
        :param qkv_bias: 计算qkv矩阵时linear是否需要添加bias
        :param qk_scale: None则使用d_k**-0.5
        :param drop_rate: pos_drop_rate and linear_drop_rate
        :param attn_drop_rate: after attn drop rate
        :param drop_path_rate:
        :param norm_layer:
        :param patch_norm:
        :param self_attn: 注意力机制是否为自注意力
        :param low_res: 是否为低分辨率输入
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.require_patch_embed = require_patch_embed
        self.down_sample = down_sample
        self.self_attn = self_attn

        patch_size_t = [patch_size[0]]
        for i in range(3):
            patch_size_t.append(patch_size[0] * (2**i))
        patch_size_h = []
        for i in range(4):
            patch_size_h.append(patch_size[1] * (i + 1))
        patch_size_w = []
        for i in range(4):
            patch_size_w.append(patch_size[2] * (i + 1))
        if configs["use_multi_scale_patch_embedding"] == 1:
            self.patch_embed = (
                MultiScalePatchEmbed3D(
                    patch_size_t=patch_size_t,
                    patch_size_h=patch_size_h,
                    patch_size_w=patch_size_w,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None,
                )
                if self.require_patch_embed
                else nn.Identity()
            )
        else:
            self.patch_embed = (
                PatchEmbed3D(
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None,
                )
                if self.require_patch_embed
                else nn.Identity()
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.pos_encoding = (
            nn.Parameter(
                torch.zeros(
                    (
                        1,
                        embed_dim,
                        configs["input_length"] // patch_size[0],
                        configs["img_height"] // patch_size[1],
                        configs["img_width"] // patch_size[2],
                    )
                ),
                requires_grad=True,
            )
            if require_patch_embed
            else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        if drop_path_rate == 0:
            dpr = None
        else:
            # stochastic depth
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depths)
            ]  # stochastic depth decay rule

        # build layer
        self.layer = BasicLayer(
            configs=configs,
            patch_size=patch_size,
            dim=embed_dim,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            self_attn=self_attn,
            low_res=low_res,
        )

        self.norm = norm_layer(embed_dim)

        if self.down_sample:
            self.down_sample_block = PatchMerging(dim=embed_dim, norm_layer=norm_layer)

    def forward(self, x, memory=None):
        if self.require_patch_embed:
            x = self.patch_embed(x)
            x += self.pos_encoding
        x = self.pos_drop(x)

        if self.self_attn:
            x = self.layer(x.contiguous())
        else:
            x = self.layer(x.contiguous(), memory)

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        if self.down_sample:
            x = rearrange(x, "n c d h w -> n d h w c")
            x_down = self.down_sample_block(x)
            x = rearrange(x, "n d h w c -> n c d h w")
            x_down = rearrange(x_down, "n d h w c -> n c d h w")
            return x, x_down
        return x
