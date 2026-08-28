from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from models.GVBF.gvbf_network import get_model as _gvbf_get_model


def _resolve_drift_channels(configs):
    channels = getattr(configs, "drift_channels", "auto")
    if channels is None or (isinstance(channels, str) and channels.lower() == "auto"):
        in_category = getattr(configs, "in_category", None)
        if in_category is not None:
            return len(in_category)
        return 1
    return int(channels)


def _pad_size(size, multiple=64):
    return ((size + multiple - 1) // multiple) * multiple


def _pad_tensor(x, multiple=64):
    _, _, height, width = x.shape
    padded_height = _pad_size(height, multiple)
    padded_width = _pad_size(width, multiple)
    if height == padded_height and width == padded_width:
        return x, None
    return F.pad(x, (0, padded_width - width, 0, padded_height - height), mode="replicate"), (height, width)


def _unpad_tensor(x, original_size):
    if original_size is None:
        return x
    height, width = original_size
    return x[:, :, :height, :width]


class DriftUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, gen_per_input=4, unet_config="auto"):
        super().__init__()
        self.in_channels = int(in_channels)
        self.gen_per_input = int(gen_per_input)
        out_channels = self.in_channels * self.gen_per_input
        if unet_config in (None, "auto"):
            unet_config = f"{self.in_channels}_{out_channels}_64"
        self.unet = _gvbf_get_model(
            SimpleNamespace(config=unet_config, weights_path=None)
        )

    def forward(self, x):
        x_padded, original_size = _pad_tensor(x)
        timestep = torch.zeros(x_padded.shape[0], device=x_padded.device)
        out = self.unet(x_padded, timestep)
        out = _unpad_tensor(out, original_size)
        batch, _, height, width = out.shape
        return out.reshape(batch, self.gen_per_input, self.in_channels, height, width)


def get_model(configs):
    return DriftUNetGenerator(
        in_channels=_resolve_drift_channels(configs),
        gen_per_input=int(getattr(configs, "drift_gen_per_input", 4)),
        unet_config=getattr(configs, "drift_unet_config", "auto"),
    )
