from diffusers import UNet2DModel
import torch
from pathlib import Path
from safetensors.torch import load_model


class MyUNet2DModel(UNet2DModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).sample


class TwoModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, t, alpha=None):
        return torch.concat(
            [self.model1(x, t, alpha), self.model2(x, t, alpha)], dim=1
        )


def _make_unet_small(in_channels, out_channels, c, cond):
    block_out_channels = (c, c, 2 * c, 2 * c, 4 * c, 4 * c)
    down_block_types = (
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D",
    )
    return MyUNet2DModel(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
        class_embed_type="timestep" if cond else None,
    )


def _make_unet(in_channels, out_channels, c, cond):
    block_out_channels = (c, c, 2 * c, 2 * c, 2 * c, 4 * c, 4 * c)
    down_block_types = (
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
    )
    return MyUNet2DModel(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
        class_embed_type="timestep" if cond else None,
    )


def _parse_config(config_str):
    class_cond = "cond" in config_str
    digits = [int(p) for p in config_str.split("_") if p.isdigit()]
    in_channels, out_channels, c = digits
    return in_channels, out_channels, c, class_cond


def get_model(model_config):
    config_str = model_config.config
    in_c, out_c, c, cond = _parse_config(config_str)

    _get_model = _make_unet_small if "small" in config_str else _make_unet

    if "two_model" in config_str:
        m1 = _get_model(in_c, out_c, c, cond)
        m2 = _get_model(in_c, out_c, c, cond)
        model = TwoModel(m1, m2)
    else:
        model = _get_model(in_c, out_c, c, cond)

    if model_config.weights_path:
        print(f"Loading model from {model_config.weights_path}")
        load_model(model, Path(model_config.weights_path))

    return model
