import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL


SOURCE_ROOT = Path(
    os.environ.get(
        "OPENSTP_PREDIFF_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[3] / "PreDiff_code" / "prediff" / "src"),
    )
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from prediff.diffusion.utils import make_beta_schedule, extract_into_tensor, noise_like  # noqa: E402
from prediff.models.cuboid_transformer_unet_diffusion import CuboidTransformerDiffusionUNetAE  # noqa: E402
from prediff.taming.vae import IdentityFirstStage  # noqa: E402


class PreDiffModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.img_size = list(configs["img_size"])
        self.pre_seq_length = int(configs["total_seq"][0])
        self.aft_seq_length = int(configs["total_seq"][1])
        self.in_channels = len(configs["in_category"])
        self.out_channels = len(configs["out_category"])
        self.label_idx = configs["label_idx"]
        self.parameterization = configs.get("prediff_parameterization", "eps")
        self.timesteps = int(configs.get("prediff_timesteps", 200))
        self.clip_denoised = bool(configs.get("prediff_clip_denoised", False))
        self.learn_logvar = bool(configs.get("prediff_learn_logvar", False))
        self.l_simple_weight = float(configs.get("prediff_l_simple_weight", 1.0))
        self.original_elbo_weight = float(configs.get("prediff_original_elbo_weight", 0.0))
        self.inference_timesteps = int(configs.get("prediff_inference_timesteps", 20))

        self.first_stage_model = self._build_first_stage_model()
        latent_in_shape, latent_out_shape = self._infer_latent_shapes()
        self.latent_in_shape = latent_in_shape
        self.latent_out_shape = latent_out_shape

        self.denoiser = CuboidTransformerDiffusionUNetAE(
            input_shape=latent_in_shape,
            target_shape=latent_out_shape,
            base_units=int(configs.get("prediff_base_units", 64)),
            scale_alpha=float(configs.get("prediff_scale_alpha", 1.0)),
            depth=list(configs.get("prediff_depth", [1, 1])),
            downsample=int(configs.get("prediff_downsample", 2)),
            downsample_type=configs.get("prediff_downsample_type", "patch_merge"),
            upsample_type=configs.get("prediff_upsample_type", "upsample"),
            upsample_kernel_size=int(configs.get("prediff_upsample_kernel_size", 3)),
            block_attn_patterns=configs.get("prediff_self_pattern", "axial"),
            num_heads=int(configs.get("prediff_num_heads", 4)),
            attn_drop=float(configs.get("prediff_attn_drop", 0.0)),
            proj_drop=float(configs.get("prediff_proj_drop", 0.0)),
            ffn_drop=float(configs.get("prediff_ffn_drop", 0.0)),
            ffn_activation=configs.get("prediff_ffn_activation", "gelu"),
            gated_ffn=bool(configs.get("prediff_gated_ffn", False)),
            norm_layer=configs.get("prediff_norm_layer", "layer_norm"),
            padding_type=configs.get("prediff_padding_type", "zeros"),
            checkpoint_level=int(configs.get("prediff_checkpoint_level", 0)),
            use_relative_pos=bool(configs.get("prediff_use_relative_pos", True)),
            self_attn_use_final_proj=bool(configs.get("prediff_self_attn_use_final_proj", True)),
            time_embed_channels_mult=int(configs.get("prediff_time_embed_channels_mult", 4)),
            time_embed_use_scale_shift_norm=bool(configs.get("prediff_time_embed_use_scale_shift_norm", False)),
            time_embed_dropout=float(configs.get("prediff_time_embed_dropout", 0.0)),
            unet_res_connect=bool(configs.get("prediff_unet_res_connect", True)),
        )

        self.register_schedule(
            beta_schedule=configs.get("prediff_beta_schedule", "linear"),
            timesteps=self.timesteps,
            linear_start=float(configs.get("prediff_linear_start", 1e-4)),
            linear_end=float(configs.get("prediff_linear_end", 2e-2)),
            cosine_s=float(configs.get("prediff_cosine_s", 8e-3)),
        )
        logvar_init = float(configs.get("prediff_logvar_init", 0.0))
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(logvar, requires_grad=True)
        else:
            self.register_buffer("logvar", logvar)

    def _build_first_stage_model(self):
        if not self.configs.get("prediff_use_vae", False):
            return IdentityFirstStage()

        out_channels = int(self.configs.get("prediff_vae_out_channels", self.out_channels))
        vae = AutoencoderKL(
            in_channels=int(self.configs.get("prediff_vae_in_channels", out_channels)),
            out_channels=out_channels,
            down_block_types=tuple(["DownEncoderBlock2D"] * len(self.configs.get("prediff_vae_block_out_channels", [64, 128, 256]))),
            up_block_types=tuple(["UpDecoderBlock2D"] * len(self.configs.get("prediff_vae_block_out_channels", [64, 128, 256]))),
            block_out_channels=tuple(self.configs.get("prediff_vae_block_out_channels", [64, 128, 256])),
            latent_channels=int(self.configs.get("prediff_vae_latent_channels", 4)),
            layers_per_block=int(self.configs.get("prediff_vae_layers_per_block", 2)),
            act_fn="silu",
            norm_num_groups=32,
        )
        ckpt_path = self.configs.get("prediff_vae_ckpt_path", None)
        if ckpt_path:
            ckpt = Path(ckpt_path)
            if not ckpt.is_absolute():
                ckpt = SOURCE_ROOT.parent.parent / ckpt
            if ckpt.exists():
                state_dict = torch.load(str(ckpt), map_location="cpu")
                vae.load_state_dict(state_dict, strict=False)
            else:
                warnings.warn(f"PreDiff VAE checkpoint not found at {ckpt}, using randomly initialized VAE.")
        else:
            warnings.warn("PreDiff VAE checkpoint not provided, using randomly initialized VAE.")
        return vae

    def _infer_latent_shapes(self):
        device = next(self.first_stage_model.parameters(), torch.zeros(1)).device
        with torch.no_grad():
            cond = torch.zeros(1, self.pre_seq_length, self.out_channels, self.img_size[0], self.img_size[1], device=device)
            target = torch.zeros(1, self.aft_seq_length, self.out_channels, self.img_size[0], self.img_size[1], device=device)
            cond_latent = self._encode_frames_ncthw(cond)
            target_latent = self._encode_frames_ncthw(target)
        return list(cond_latent.shape[1:]), list(target_latent.shape[1:])

    def _select_condition_channels(self, input_seq):
        if input_seq.shape[2] == self.out_channels:
            return input_seq
        start, end = self.label_idx
        return input_seq[:, :, start:end, :, :]

    @staticmethod
    def _to_nthwc(x):
        return x.permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def _to_ncthw(x):
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _encode_frames_ncthw(self, x):
        if isinstance(self.first_stage_model, IdentityFirstStage):
            return self._to_nthwc(x)
        batch, seq, channels, height, width = x.shape
        spatial = x.reshape(batch * seq, channels, height, width)
        encoded = self.first_stage_model.encode(spatial)
        if hasattr(encoded, "latent_dist"):
            latent = encoded.latent_dist.sample()
        elif hasattr(encoded, "sample") and callable(encoded.sample):
            latent = encoded.sample()
        else:
            latent = encoded
        latent = latent * float(self.configs.get("prediff_vae_scale_factor", 1.0))
        latent = latent.reshape(batch, seq, latent.shape[1], latent.shape[2], latent.shape[3])
        return self._to_nthwc(latent)

    def _decode_frames_nthwc(self, z):
        if isinstance(self.first_stage_model, IdentityFirstStage):
            return self._to_ncthw(z)
        batch, seq, height, width, channels = z.shape
        spatial = z.permute(0, 1, 4, 2, 3).reshape(batch * seq, channels, height, width)
        spatial = spatial / float(self.configs.get("prediff_vae_scale_factor", 1.0))
        decoded = self.first_stage_model.decode(spatial)
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        decoded = decoded.reshape(batch, seq, decoded.shape[1], decoded.shape[2], decoded.shape[3])
        return decoded.contiguous()

    def persistence_predict(self, input_seq, steps):
        cond = self._select_condition_channels(input_seq)
        last = cond[:, -1:, :, :, :]
        return last.repeat(1, steps, 1, 1, 1)

    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.num_timesteps = int(timesteps)

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1.0)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        else:
            lvlb_weights = 0.5 * torch.sqrt(self.alphas_cumprod) / (2.0 - self.alphas_cumprod)
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_losses(self, x_start, cond, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.denoiser(x_noisy, t, cond)
        target = noise if self.parameterization == "eps" else x_start
        loss_simple = torch.nn.functional.mse_loss(model_output, target, reduction="none").mean(dim=(1, 2, 3, 4))
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = (self.lvlb_weights[t] * loss_simple).mean()
        loss = loss + self.original_elbo_weight * loss_vlb
        return loss

    def training_loss(self, input_seq, target_seq):
        cond_seq = self._select_condition_channels(input_seq)
        cond_latent = self._encode_frames_ncthw(cond_seq)
        target_latent = self._encode_frames_ncthw(target_seq)
        t = torch.randint(0, self.num_timesteps, (target_latent.shape[0],), device=target_latent.device).long()
        return self.p_losses(target_latent, cond_latent, t)

    def p_mean_variance(self, zt, cond, t):
        model_out = self.denoiser(zt, t, cond)
        z_recon = self.predict_start_from_noise(zt, t=t, noise=model_out) if self.parameterization == "eps" else model_out
        if self.clip_denoised:
            z_recon = z_recon.clamp(-1.0, 1.0)
        return self.q_posterior(x_start=z_recon, x_t=zt, t=t)

    @torch.no_grad()
    def p_sample(self, zt, cond, t):
        model_mean, _, model_log_variance = self.p_mean_variance(zt=zt, cond=cond, t=t)
        noise = noise_like(zt.shape, zt.device)
        mask_shape = [1] * zt.ndim
        mask_shape[0] = zt.shape[0]
        nonzero_mask = (1 - (t == 0).float()).reshape(*mask_shape)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, input_seq, steps=None, sample_steps=None):
        steps = self.aft_seq_length if steps is None else steps
        if steps != self.aft_seq_length:
            raise ValueError(f"PreDiff currently expects fixed output length {self.aft_seq_length}, got {steps}.")
        cond_seq = self._select_condition_channels(input_seq)
        cond_latent = self._encode_frames_ncthw(cond_seq)
        latent_shape = (input_seq.shape[0], *self.latent_out_shape)
        sample_steps = self.inference_timesteps if sample_steps is None else sample_steps
        z = torch.randn(latent_shape, device=input_seq.device)
        for i in reversed(range(sample_steps)):
            t = torch.full((latent_shape[0],), int(i * self.num_timesteps / sample_steps), device=input_seq.device, dtype=torch.long)
            z = self.p_sample(z, cond_latent, t)
        return self._decode_frames_nthwc(z)


class Main(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.configs = configs
        self.device = device
        self.pre_seq_length = int(configs["total_seq"][0])
        self.aft_seq_length = int(configs["total_seq"][1])
        self.model = PreDiffModel(configs)

    def _prepare_input_seq(self, batch_x):
        batch_size, seq_len, channels, height, width = batch_x.shape
        if seq_len >= self.pre_seq_length:
            return batch_x[:, -self.pre_seq_length :]
        padding = torch.zeros(batch_size, self.pre_seq_length - seq_len, channels, height, width, device=batch_x.device, dtype=batch_x.dtype)
        return torch.cat([padding, batch_x], dim=1)

    def predict(self, batch_x, steps=None, sample_steps=None):
        batch_x = self._prepare_input_seq(batch_x)
        return self.model.sample(batch_x, steps=steps, sample_steps=sample_steps)

    def forward(self, batch_x, batch_y=None):
        steps = self.aft_seq_length
        if batch_y is not None and torch.is_tensor(batch_y) and batch_y.dim() >= 2:
            steps = batch_y.shape[1]
        return self.predict(batch_x, steps=steps)
