class Drift_Config:
    def __init__(self, config_name=None):
        self.model_config = self.conf_choise(config_name)

    def conf_choise(self, name):
        if name in (None, "default"):
            config = {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "drift_channels": "auto",
                "drift_unet_config": "auto",
                "drift_gen_per_input": 4,
                "drift_pos_bank_size": 64,
                "drift_neg_bank_size": 128,
                "drift_pos_samples": 4,
                "drift_neg_samples": 4,
                "drift_negative_weight": 1.0,
                "drift_max_tokens": 1024,
                "drift_inference_reduce": "mean",
            }
        elif name == "minimal":
            config = {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "drift_channels": "auto",
                "drift_unet_config": "auto",
                "drift_gen_per_input": 4,
                "drift_pos_bank_size": 16,
                "drift_neg_bank_size": 32,
                "drift_pos_samples": 2,
                "drift_neg_samples": 2,
                "drift_negative_weight": 1.0,
                "drift_max_tokens": 512,
                "drift_inference_reduce": "mean",
            }
        else:
            raise ValueError("Unknown Drift config: " + str(name))

        return config
