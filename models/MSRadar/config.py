class MS_RadarFormer_Config:
    def __init__(self):
        self.model_config = {
            "patch_size": 4,    
            "model_patch_size": [ 3, 4, 4 ],
            "window_size": [ 3, 4, 4 ],
            "embed_dim": 256,
            "num_heads": 16,
            "depths": 6,    
            "attn_drop_rate": 0.0,
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "use_multi_resolution_branch": 0,
            "use_multi_scale_patch_embedding": 0,

            "learning_rate": 1e-4,
            "test_interval": 1,
            "weight_decay": 1e-5,
            "pct_start": 0.3,
            "std_method": "z_score",
        }
