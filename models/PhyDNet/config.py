class PhyDNet_Config:
    def __init__(self):
        self.model_config = {
            "patch_size": 2,
            "learning_rate": 1e-04,
            "initial_lr": 1e-02,
            "lr_min": 1e-06,
            "batch_size": 16,
            "drop_path" : 0.1,
            "drop":0.0,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "k_decay": 1.0,
            "scheduler": "cosine",
            "warmup_t" : 0, 
            "warmup_lr_init": 1e-05,
            "warmup_epoch": 0,
            "loss_function": "mse",
        }