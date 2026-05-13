import argparse



def config_update(configs, new_data):
    for k, v in new_data.items():
        setattr(configs, k, v)  
    
    return configs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/SimVP_gSTA.yaml')
    args = parser.parse_args()
    new = {
        "lr": 1e-3,
        "batch_size": 16,
        "save_dir": "./outputs"
        }

    configs = config_update(args, new)
    print(configs)
