import json
import os


class PreDiff_Config:
    def __init__(self, config_source=None):
        self.model_config = {
            "prediff_use_vae": False,
            "prediff_vae_ckpt_path": None,
            "prediff_vae_scale_factor": 1.0,
            "prediff_vae_in_channels": None,
            "prediff_vae_out_channels": None,
            "prediff_vae_block_out_channels": [64, 128, 256],
            "prediff_vae_latent_channels": 4,
            "prediff_vae_layers_per_block": 2,
            "prediff_base_units": 64,
            "prediff_scale_alpha": 1.0,
            "prediff_depth": [1, 1],
            "prediff_downsample": 2,
            "prediff_downsample_type": "patch_merge",
            "prediff_upsample_type": "upsample",
            "prediff_upsample_kernel_size": 3,
            "prediff_self_pattern": "axial",
            "prediff_num_heads": 4,
            "prediff_attn_drop": 0.0,
            "prediff_proj_drop": 0.0,
            "prediff_ffn_drop": 0.0,
            "prediff_ffn_activation": "gelu",
            "prediff_gated_ffn": False,
            "prediff_norm_layer": "layer_norm",
            "prediff_padding_type": "zeros",
            "prediff_checkpoint_level": 0,
            "prediff_use_relative_pos": True,
            "prediff_self_attn_use_final_proj": True,
            "prediff_time_embed_channels_mult": 4,
            "prediff_time_embed_use_scale_shift_norm": False,
            "prediff_time_embed_dropout": 0.0,
            "prediff_unet_res_connect": True,
            "prediff_timesteps": 200,
            "prediff_beta_schedule": "linear",
            "prediff_linear_start": 1e-4,
            "prediff_linear_end": 2e-2,
            "prediff_cosine_s": 8e-3,
            "prediff_parameterization": "eps",
            "prediff_learn_logvar": False,
            "prediff_logvar_init": 0.0,
            "prediff_l_simple_weight": 1.0,
            "prediff_original_elbo_weight": 0.0,
            "prediff_inference_timesteps": 20,
            "prediff_metric_sampling_steps": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": "cosineannealing",
            "scheduler_T_max": 300,
            "loss_function": "prediff",
        }
        self._apply_config_source(config_source)

    def _apply_config_source(self, config_source):
        if config_source is None:
            return
        if isinstance(config_source, dict):
            self.model_config.update(config_source)
            return
        if not isinstance(config_source, str):
            raise ValueError(f"不支持的 PreDiff 配置来源: {type(config_source).__name__}")

        config_path = os.path.join(os.path.dirname(__file__), "configs", f"{config_source}.json")
        if not os.path.exists(config_path):
            raise ValueError(f"未找到 PreDiff 模型配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.model_config.update(json.load(f))
