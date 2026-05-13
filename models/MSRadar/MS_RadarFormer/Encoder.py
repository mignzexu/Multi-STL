import torch.nn as nn

from .SpatialTemporalAttention import SpatialTemporalAttention


class Encoder(nn.Module):
    def __init__(
        self,
        configs,
        patch_size=(4, 5, 5),
        in_chans=1,
        embed_dim=256,
        depths=8,
        num_heads=16,
        window_size=(4, 4, 4),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    ):
        super().__init__()
        self.configs = configs
        self.first_transformer_block = SpatialTemporalAttention(
            configs=configs,
            patch_size=patch_size,
            in_chans=in_chans,
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depths=1,
            require_patch_embed=True,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            self_attn=True,
            down_sample=True,
        )

        self.original_resolution_block = SpatialTemporalAttention(
            configs=configs,
            patch_size=patch_size,
            in_chans=in_chans,
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depths=depths,
            require_patch_embed=False,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            self_attn=True,
            down_sample=False,
        )

        if configs["use_multi_resolution_branch"] == 1:
            self.low_resolution_block = SpatialTemporalAttention(
                configs=configs,
                patch_size=patch_size,
                in_chans=in_chans,
                window_size=window_size,
                embed_dim=embed_dim * 2,
                num_heads=num_heads,
                depths=depths,
                require_patch_embed=False,
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                self_attn=True,
                down_sample=False,
                low_res=True,
            )

    def forward(self, x):
        x, x_low_res = self.first_transformer_block(x)
        x = self.original_resolution_block(x)
        if self.configs["use_multi_resolution_branch"] == 1:
            x_low_res = self.low_resolution_block(x_low_res)
            return x, x_low_res
        return x, None
