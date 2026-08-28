import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from .model_utils import PatchEmbedBack3D, ConvOut
import argparse
from einops import rearrange

try:
    from ...Model_system import distribute_model_layers
except ImportError:
    from models.Model_system import distribute_model_layers


class MsRadarFormer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = self.config_transform(configs)
        self.encoder = Encoder(
            configs=self.configs,
            patch_size=self.configs["model_patch_size"],
            in_chans=self.configs["img_channel"] * (self.configs["patch_size"] ** 2),
            embed_dim=self.configs["embed_dim"],
            depths=self.configs["depths"],
            num_heads=self.configs["num_heads"],
            window_size=self.configs["window_size"],
            drop_rate=self.configs["drop_rate"],
            attn_drop_rate=self.configs["attn_drop_rate"],
            drop_path_rate=self.configs["drop_path_rate"],
        )
        self.decoder = Decoder(
            configs=self.configs,
            patch_size=self.configs["model_patch_size"],
            in_chans=self.configs["img_channel"] * (self.configs["patch_size"] ** 2),
            embed_dim=self.configs["embed_dim"],
            depths=self.configs["depths"],
            num_heads=self.configs["num_heads"],
            window_size=self.configs["window_size"],
            drop_rate=self.configs["drop_rate"],
            attn_drop_rate=self.configs["attn_drop_rate"],
            drop_path_rate=self.configs["drop_path_rate"],
        )
        self.patch_embed_back = PatchEmbedBack3D(
            patch_size=self.configs["model_patch_size"],
            # in_chans为输出channel
            in_chans=self.configs["embed_dim"]
            if self.configs["use_multi_resolution_branch"] == 1
            else self.configs["embed_dim"] // 2,
            embed_dim= self.configs["embed_dim"] * 2  # embed_dim为输入channel
            if self.configs["use_multi_resolution_branch"] == 1
            else self.configs["embed_dim"],
        )
        self.conv_out = ConvOut(
            self.configs,
            self.configs["embed_dim"]
            if self.configs["use_multi_resolution_branch"] == 1
            else self.configs["embed_dim"] // 2,
            self.configs["img_out_channel"] * (self.configs["patch_size"] ** 2),
        )
        self._layer_devices = None

    def _get_layer_groups(self, devices):
        main_dev, middle_dev = devices[0], devices[1]
        return [
            (self.encoder, main_dev),
            (self.decoder, middle_dev),
            (self.patch_embed_back, main_dev),
            (self.conv_out, main_dev),
        ]

    def config_transform(self, configs):
        model_config = vars(configs)
        model_config["input_length"] = configs.total_seq[0]
        model_config["output_length"] = configs.total_seq[1]
        model_config["total_length"] = sum(configs.total_seq)
        model_config["img_height"] = configs.img_size[0] // configs.patch_size
        model_config["img_width"] = configs.img_size[1] // configs.patch_size
        model_config["img_channel"] = len(configs.in_category)
        model_config["img_out_channel"] = len(configs.out_category)
        return model_config

    def patchify(self, x):
        patch_size = self.configs["patch_size"]
        if patch_size == 1:
            return x

        b, t, c, h, w = x.shape
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"Input spatial size {(h, w)} must be divisible by patch_size={patch_size}."
            )

        return rearrange(
            x,
            "b t c (ph p1) (pw p2) -> b t (c p1 p2) ph pw",
            p1=patch_size,
            p2=patch_size,
        )

    def unpatchify(self, x):
        patch_size = self.configs["patch_size"]
        if patch_size == 1:
            return x

        b, t, c, h, w = x.shape
        patch_area = patch_size * patch_size
        if c % patch_area != 0:
            raise ValueError(
                f"Channel size {c} must be divisible by patch area {patch_area}."
            )

        return rearrange(
            x,
            "b t (c p1 p2) ph pw -> b t c (ph p1) (pw p2)",
            p1=patch_size,
            p2=patch_size,
        )

    def forward(self, x):
        distribute_model_layers(self, self.configs, x.device)
        main_dev = self._layer_devices[0] if self._layer_devices else x.device
        middle_dev = self._layer_devices[1] if self._layer_devices else x.device
        x = x.to(main_dev, non_blocking=True)
        x = self.patchify(x)
        x = rearrange(x, "b t c h w -> b c t h w")
        memory, memory_low_res = self.encoder(x)
        memory = memory.to(middle_dev, non_blocking=True)
        memory_low_res = memory_low_res.to(middle_dev, non_blocking=True)
        out = self.decoder(memory, memory_low_res)
        out = out.to(main_dev, non_blocking=True)
        out = self.patch_embed_back(out)
        out = self.conv_out(out)
        out = rearrange(out, "b c t h w -> b t c h w")
        out = self.unpatchify(out)
        return out
