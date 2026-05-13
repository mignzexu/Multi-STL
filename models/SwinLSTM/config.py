class SwinLSTM_Config:
    def __init__(self):
        self.model_config = {
            "depths_downsample": '2, 6',  # 转换为列表
            "depths_upsample": '6, 2',    # 转换为列表
            "num_heads": '4, 8',          # 转换为列表
            "patch_size": 2,
            "window_size": 4,
            "embed_dim": 128,
            "batch_size": 16,
            "scheduler": "onecycle",
            "learning_rate": 1e-04,
            "initial_lr": 1e-02,
            "lr_min": 1e-06,
            "drop_path" : 0.1,
            "drop":0.0,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "k_decay": 1.0,
            "warmup_t" : 0, 
            "warmup_lr_init": 1e-05,
            "warmup_epoch": 0,
            "pct_start": 0.3,
            "loss_function": "mse",
        }