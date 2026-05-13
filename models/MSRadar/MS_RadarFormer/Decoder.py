import torch
import torch.nn as nn
from einops import rearrange

from .SpatialTemporalAttention import SpatialTemporalAttention
from .model_utils import PatchExpanding


class Decoder(nn.Module):
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
            self_attn=False,
        )

        if configs["use_multi_resolution_branch"] == 1:
            self.low_resolution_transformer = SpatialTemporalAttention(
                configs=configs,
                in_chans=in_chans,
                patch_size=patch_size,
                window_size=window_size,
                embed_dim=embed_dim * 2,
                num_heads=num_heads,
                depths=depths,
                require_patch_embed=False,
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                self_attn=False,
                low_res=True,
            )
            self.up_sample = PatchExpanding(embed_dim * 2)

        self.out_transformer_block = SpatialTemporalAttention(
            configs=configs,
            in_chans=in_chans,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim * 2
            if configs["use_multi_resolution_branch"] == 1
            else embed_dim,
            num_heads=num_heads,
            depths=depths // 2,
            require_patch_embed=False,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            self_attn=True,
        )

        # 通过register_buffer()登记过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新
        self.register_buffer(
            "input",
            torch.zeros(
                (
                    configs["batch_size"],
                    configs["img_channel"] * (configs["patch_size"] ** 2),
                    configs["input_length"],
                    configs["img_height"],
                    configs["img_width"],
                )
            ),
        )

        # self.input = torch.zeros(
        #     (
        #         configs["batch_size"],
        #         configs["img_channel"],
        #         configs["input_length"],
        #         configs["img_height"],
        #         configs["img_width"],
        #     )
        # ).to(configs["device"])

    def forward(self, origin_res_memory, low_res_memory):
        out, out_low_res = self.first_transformer_block(self.input)
        out = self.original_resolution_block(out, origin_res_memory)

        if self.configs["use_multi_resolution_branch"] == 1:
            out_low_res = self.low_resolution_transformer(out_low_res, low_res_memory)
            out_low_res = rearrange(out_low_res, "n c d h w -> n d h w c")
            out_low_res = self.up_sample(out_low_res)
            out_low_res = rearrange(out_low_res, "n d h w c -> n c d h w")

            out = torch.concat([out, out_low_res], dim=1)
        out = self.out_transformer_block(out)
        return out
