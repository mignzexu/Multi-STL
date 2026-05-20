import os
import json
import argparse
import shutil


def create_args():
    parser = argparse.ArgumentParser(description='Multi-STL: 多源数据融合降水临近预报模型')

    # 基本配置
    parser.add_argument('--ex_name', '-ex', type=str, default='test', help='本次实验的名称')
    parser.add_argument('--load_model', '-mod', type=str, default='gsta', help='模型选择')
    parser.add_argument('--dataset', '-ds', type=str, default='weatherbench', choices=[
                                'SDweather', 'weatherbench', 'sevir'
                                ], help='数据集选择')

    #运行模式
    parser.add_argument('--train', action='store_true', default=False, help='训练')
    parser.add_argument('--retrain', action='store_true', default=False, help='再训练')
    parser.add_argument('--test', action='store_true', default=False, help='测试')

    # 训练和硬件配置
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='批次大小')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='训练轮数')
    parser.add_argument('--save_grad', default=None, help='是否保存梯度?')
    parser.add_argument('--save_config', "-sc", action='store_true', default=False, help='是否保存配置文件?')
    parser.add_argument('--device', type=str, default='all', help='使用的GPU ID,多个GPU用逗号隔开,使用"all"表示使用所有可用GPU,使用"cpu"表示使用CPU')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pin_memory', default=None, help='是否使用 pinned memory 加速数据加载')

    # 选择配置
    parser.add_argument('--data_config',"-dc", type=str, default= "test", help='数据集配置文件')
    parser.add_argument('--model_config',"-mc", type=str, default= None, help='模型配置文件')

    #测试指令
    parser.add_argument('--save', action='store_true', default=False, help='是否保存预测图')

    # 文件路径和保存
    parser.add_argument('--work_dirs', '-wd', type=str, default='work_dirs', help='模型保存路径（默认包含时间戳）') 
    parser.add_argument('--data_dir', type=str, default="/scratch/mingze/data", help='数据集总路径，内部请按照ReadMe中配置')
    parser.add_argument('--save_mode', type=str, default="manual", choices=["auto", "manual"], help='数据集总路径，内部请按照ReadMe中配置')
    parser.add_argument('--test_interval', type=int, default=1, help='测试间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='保存间隔')

    #多卡
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader worker 数量，mmap 数据建议先用 0')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='num_workers > 0 时每个 worker 的预取 batch 数')
    parser.add_argument('--persistent_workers', default=True, help='是否保持 DataLoader worker 常驻')

    parser.add_argument(
        '--strategy',
        '-sty',
        type=str,
        default='auto',
        choices=[
            'auto',
            'ddp',
            'ddp_find_unused_parameters_true',
            'fsdp',
            'deepspeed_stage_2',
            'deepspeed_stage_3',
        ],
        help='Lightning 训练策略：单卡用 auto，多卡默认 ddp，大模型可尝试 fsdp'
    )

    parser.add_argument(
        '--precision',
        type=str,
        default='32-true',
        choices=['32-true', '16-mixed', 'bf16-mixed'],
        help='训练精度'
    )

    parser.add_argument(
        '--accumulate_grad_batches',
        type=int,
        default=1,
        help='梯度累积步数'
    )
    
    configs = parser.parse_args()

    return configs

def is_dist_child_process():
    """
    Lightning DDP 子进程通常会带 LOCAL_RANK / RANK 等环境变量。
    父进程没有这些变量。
    """
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def is_global_zero():
    return int(os.environ.get("RANK", "0")) == 0



def train(configs):
        
    print(">>>>>>>>>>>>>>>训练模式<<<<<<<<<<<<<<<")

    print(f"创建工程{configs.ex_name}")
    obj_dir = os.path.join(configs.work_dirs, configs.ex_name)
    configs.obj_dir = obj_dir
    from utils import train_config, save_loger
    if configs.save_config :
        config_log = train_config(
            int(configs.gpu_id), 
            mode = configs.save_mode)
    else :
        config_log = None
    if not is_dist_child_process():
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        else:
            if input(f"工程 {configs.ex_name} 已存在，是否替换？(y/n)") == "y":
                shutil.rmtree(obj_dir)
                os.makedirs(obj_dir)
            else:
                raise ValueError(f"工程 {configs.ex_name} 已存在，请 retrain 或更换工程名")
    else:
        os.makedirs(obj_dir, exist_ok=True)
    data_loader = Dataset_Instrument(configs)
    print("加载训练数据集...")
    data_loader.load_dataset(mode='train')
    train_dataset = data_loader.train_data
    print("训练数据集预加载完成\n")

    print("加载验证数据集...")
    data_loader.load_dataset(mode='valid')
    valid_dataset = data_loader.valid_data
    print("验证数据集预加载完成\n")

    print(f"创建{configs.load_model}模型......")
    model_loader = Model_Instrument(configs, mode='train')
    model = model_loader.model
    print(f"模型{configs.load_model}创建完成\n")
    
    with open(os.path.join(obj_dir, "obj_config.json"), "w", encoding="utf-8") as f:
        save_configs = vars(configs)
        json.dump(save_configs, f, ensure_ascii=False, indent=4) 

    print("配置文件保存成功\n")

    print("===============开始训练===============")

    trainer = Trainer(configs, model, train_dataset, valid_dataset)
    save_loger(config_log)
    trainer.train()

    print("===============训练完成===============")


def test(configs):

    obj_dir = os.path.join(configs.work_dirs, configs.ex_name)
    print(">>>>>>>>>>>>>>>测试模式<<<<<<<<<<<<<<<")

    print(f"加载工程配置文件...")
    config_file = os.path.join(obj_dir, "obj_config.json")

    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            test_configs = json.load(f)
        test_configs = argparse.Namespace(**test_configs)
        print("工程配置文件加载成功\n")
        
    else:
        raise ValueError(f"工程{configs.ex_name}不存在,请先训练或更换工程名")

    data_loader = Dataset_Instrument(test_configs)

    print("加载测试数据集...")
    data_loader.load_dataset(mode='test')
    test_dataset = data_loader.test_data
    print("测试数据集预加载完成\n")

    print(f"创建{test_configs.load_model}模型......")
    model_loader = Model_Instrument(test_configs, mode='test')
    model = model_loader.model
    print(f"模型{test_configs.load_model}创建完成\n")

    print("===============开始测试===============")

    tester = Tester(test_configs, model, configs.save, test_dataset)
    tester.test()

    print("===============测试完成===============")



if __name__ == '__main__':

    configs = create_args()
    
    raw_device = configs.device

    # 这一步必须在 import torch 之前设置
    if raw_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif raw_device != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = raw_device
    # raw_device == "all" 时不设置，默认使用所有可见 GPU

    import torch
    from lightning.pytorch import seed_everything

    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available() and raw_device != "cpu":
        visible_gpu_count = torch.cuda.device_count()

        configs.accelerator = "gpu"
        configs.gpu_count = int(visible_gpu_count)
        configs.devices = int(visible_gpu_count)

        # 兼容你旧代码中可能使用 configs.device 的地方
        configs.device = "gpu"

        if raw_device == "all":
            configs.gpu_id = "all"
            print(f"使用所有可见 GPU: {configs.gpu_count} 张")
        else:
            configs.gpu_id = raw_device
            print(f"使用 GPU: {raw_device}，可见 GPU 数量: {configs.gpu_count}")

    else:
        configs.accelerator = "cpu"
        configs.gpu_count = 0
        configs.devices = 1

        # 兼容旧代码
        configs.device = "cpu"
        configs.gpu_id = "cpu"

        print("使用 CPU")

    if configs.accelerator == "gpu" and configs.gpu_count > 1:
        if configs.strategy == "auto":
            print(f"检测到 {configs.gpu_count} 张 GPU，strategy=auto，将在 Trainer 中使用 ddp")
        else:
            print(f"检测到 {configs.gpu_count} 张 GPU，使用 strategy={configs.strategy}")
    elif configs.accelerator == "gpu":
        print("检测到单张 GPU，将使用单卡训练")
    else:
        print("未使用 GPU，将使用 CPU 训练")

    seed_everything(configs.seed, workers=True)
    

    from Instrument.models import Model_Instrument
    from Instrument.datasets import Dataset_Instrument
    from train_test import Trainer, Tester

    if not (configs.train or configs.retrain or configs.test) :
        raise ValueError("至少需要加载训练集或测试集")

    if configs.train :
        train(configs)

    if configs.test :
        test(configs)
    
    
