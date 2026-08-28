class GVBF_Config:
    def __init__(self, config_name=None):
        self.model_config = self.conf_choise(config_name)

    def conf_choise(self, name):
        if name == "flow":
            config = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "gvbf_weights_path": None,
                "gvbf_max_size": 0,
                "gvbf_mode": "flow",
                "gvbf_model_config": "1_1_64",
                "gvbf_solver_atol": 1e-2,
                "gvbf_solver_rtol": 1e-2,
                "gvbf_solver_dt0": None,
            }
        elif name == "biflow":
            config = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "gvbf_weights_path": None,
                "gvbf_max_size": 0,
                "gvbf_mode": "biflow",
                "gvbf_model_config": "1_2_64_cond",
                "gvbf_noise_level": 0.1,
                "gvbf_solver_atol": 1e-2,
                "gvbf_solver_rtol": 1e-2,
                "gvbf_solver_dt0": 0.1,
            }
        elif name == "condiff":
            config = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "gvbf_weights_path": None,
                "gvbf_max_size": 0,
                "gvbf_mode": "condiff",
                "gvbf_model_config": "2_1_64",
                "gvbf_solver_atol": 1e-2,
                "gvbf_solver_rtol": 1e-2,
                "gvbf_solver_dt0": None,
            }
        else:
            raise ValueError("Unknown GVBF config: " + str(name))

        return config
