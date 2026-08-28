class SBFlow_Config:
    def __init__(self, config_name=None):
        self.model_config = self.conf_choise(config_name)

    def conf_choise(self, name):
        if name is None or name == "default" or name == "lite":
            config = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "sbflow_weights_path": None,
                "sbflow_max_size": 0,
                "sbflow_model_config": "1_1_64",
                "sbflow_sigma": 1.0,
                "sbflow_t_eps": 1e-4,
                "sbflow_use_ot": False,
                "sbflow_ot_method": "sinkhorn",
                "sbflow_ot_replace": True,
                "sbflow_ot_num_threads": 1,
                "sbflow_solver_atol": 1e-2,
                "sbflow_solver_rtol": 1e-2,
                "sbflow_solver_dt0": None,
            }
        elif name == "ot" or name == "sinkhorn":
            config = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "gradient_clip_val": 1.0,
                "sbflow_weights_path": None,
                "sbflow_max_size": 0,
                "sbflow_model_config": "1_1_64",
                "sbflow_sigma": 1.0,
                "sbflow_t_eps": 1e-4,
                "sbflow_use_ot": True,
                "sbflow_ot_method": "sinkhorn",
                "sbflow_ot_replace": True,
                "sbflow_ot_num_threads": 1,
                "sbflow_solver_atol": 1e-2,
                "sbflow_solver_rtol": 1e-2,
                "sbflow_solver_dt0": None,
            }
        else:
            raise ValueError("Unknown SBFlow config: " + str(name))

        return config
