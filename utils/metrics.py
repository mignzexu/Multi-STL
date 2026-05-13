import os
import json
import argparse
import torch
from datetime import datetime
import numpy as np



class Recorder(object):  
    def __init__(self, configs):
        self.configs = configs
        self.threshold = self.configs.threshold
        self.category = self.configs.out_category
        self.current_epoch = 0
        self.file = {}

        if len(self.threshold) != len(self.category):
            raise ValueError("阈值种类与类别数量不一致")
        
        self.metrics = self.configs.metrics
        self.convention_epoch = []
        self.cvm_epoch = []
        self.historical_metrics = {}
        self.test_metrics = {}

    def register_epoch(self, epoch):
        """创建每个epoch的指标记录结构"""
        self.historical_metrics[str(epoch)] = {
            "time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train": {},
            "valid": {}
        }
        self.current_epoch = epoch

    def train_step(self, loss, lr):
        """记录训练过程中的损失和学习率"""
        self.historical_metrics[str(self.current_epoch)]["train"]["loss"] = f"{loss:.6f}"
        self.historical_metrics[str(self.current_epoch)]["train"]["lr"] = f"{lr:.4e}"

    def valid_step(self, loss):
        """记录验证过程中的损失"""
        self.historical_metrics[str(self.current_epoch)]["valid"]["loss"] = f"{loss:.6f}"
        metrics = self.metrics_statistics()
        self.historical_metrics[str(self.current_epoch)]["valid"].update(metrics)

    def test_step(self):
        """记录测试指标"""
        metrics = self.metrics_statistics()
        self.test_metrics.update(metrics)
        self.save_result()
        

    def metrics_statistics(self):
        """计算并记录每个epoch的指标平均值"""
        #self.single_epoch : [batch_num, metrics_num, cate_num]
        cal_metrics = {}

        for c in range(len(self.category)):
            cal_metrics[self.category[c]] = {}
            for m in range(len(self.metrics)):
                if self.metrics[m] == 'cvm':
                    category_cvm = [batch[c] for batch in self.cvm_epoch]
                    static = np.array(category_cvm, dtype=float) 
                    cal_metrics[self.category[c]]["csi"] = np.round(np.mean(static[:, 0], axis=0), 6).tolist()
                    cal_metrics[self.category[c]]["pod"] = np.round(np.mean(static[:, 1], axis=0), 6).tolist()
                    cal_metrics[self.category[c]]["far"] = np.round(np.mean(static[:, 2], axis=0), 6).tolist()
                    cal_metrics[self.category[c]]["hss"] = np.round(np.mean(static[:, 3], axis=0), 6).tolist()
                else:
                    static = np.array(self.convention_epoch) 
                    cal_metrics[self.category[c]][self.metrics[m]] = np.mean(static[:, m, c], axis=0)  

        self.convention_epoch.clear()
        self.cvm_epoch.clear()

        return cal_metrics

    def save_process(self):
        """保存训练和验证过程中的指标"""
        save_dir = self.configs.obj_dir
        save_path = os.path.join(save_dir, "metrics.json")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.historical_metrics, f, ensure_ascii=False, indent=4) 
        except Exception as e:
            print(f"指标保存失败\n")
            print(e)
    
    def save_result(self):
        """保存测试结果相关指标"""
        save_dir = self.configs.obj_dir
        save_path = os.path.join(save_dir, "result.json")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.test_metrics, f, ensure_ascii=False, indent=4) 
        except Exception as e:
            print(f"指标保存失败\n")
            print(e)

    def load_process(self):
        """加载训练或测试过程中的指标, 用于继续训练"""
        save_dir = os.path.join(self.configs.obj_dir, "process")
        save_path = os.path.join(save_dir, "metrics.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                self.historical_metrics = json.load(f)

    
    @torch.no_grad()
    def within_the_epoch(self, pred, label): # [B, T, C, H, W]
        """计算单个batch的指标并记录"""
        # pred = self.standardizer.de_standardizing(pred)
        # label = self.standardizer.de_standardizing(label)
        
        batch_convention = []
        cate = len(self.category)
        for met in self.metrics:

            if met == 'mae':
                mae_list = []
                for i in range(cate):
                    mae_list.append(self.mae(pred[:, :, i, :, :], label[:, :, i, :, :]))
                batch_convention.append(mae_list)

            elif met == 'mse':
                mse_list = []
                for i in range(cate):
                    mse_list.append(self.mse(pred[:, :, i, :, :], label[:, :, i, :, :]))
                batch_convention.append(mse_list)

            elif met == 'rmse':
                rmse_list = []
                for i in range(cate):
                    rmse_list.append(self.rmse(pred[:, :, i, :, :], label[:, :, i, :, :]))
                batch_convention.append(rmse_list)
            
            elif met == "cvm":
                cvm_list = []
                for i in range(cate):
                    cvm_list.append(self.cvm(pred[:, :, i, :, :], label[:, :, i, :, :], self.threshold[i]))
                self.cvm_epoch.append(cvm_list) 

            else:
                raise NotImplementedError(f'指标 {met} 未实现')
        
        self.convention_epoch.append(batch_convention) #[metrics_num, cate_num]

    @staticmethod
    def mae(pred, true): #[b, t, c, h, w]
        return torch.mean(torch.abs(pred - true)).item()

    @staticmethod
    def mse(pred, true):
        return torch.mean((pred - true) ** 2).item()
    
    @staticmethod
    def rmse(pred, true):
        return torch.sqrt(torch.mean((pred - true) ** 2)).item()

    @staticmethod
    def cvm(pred, true, threshold): 
        """Categorical Verification Metrics"""
        metrics = [[], [], [], []]
        for k in threshold :
            tp = torch.sum((pred >= k) & (true >= k)).item()
            fp = torch.sum((pred >= k) & (true < k)).item()
            fn = torch.sum((pred < k) & (true >= k)).item()
            tn = torch.sum((pred < k) & (true < k)).item()
            metrics[0].append(tp / (tp + fn + fp + 1e-8)) # csi
            metrics[1].append(tp / (tp + fn + 1e-8)) # pod
            metrics[2].append(fp / (tp + fp + 1e-8)) # far
            metrics[3].append((2 * (tp * tn - fp * fn)) / ((tp + fn) * (tn + fn) + (tp + fp) * (tn + fp)+ 1e-8)) # hss

        return metrics #[metric, threshold]




if __name__ == "__main__":

    configs = argparse.Namespace(
        threshold=[[0.5, 0.75, 1.0], [0.5]],
        out_category=["tp", "w10"],
        metrics=["mae", "mse", "rmse", "cvm"],
        obj_dir="./save"
    )
    
    if not os.path.exists(configs.obj_dir):
        os.makedirs(configs.obj_dir)

    recorder = Recorder(configs)
    pred = torch.randn(1, 5, 2, 16, 32)
    label = torch.randn(1, 5, 2, 16, 32)

    recorder.register_epoch(1)
    recorder.within_the_epoch(pred, label)
    recorder.train_step(0.25, 5e-4)
    recorder.valid_step(0.125)

    recorder.register_epoch(2)
    recorder.within_the_epoch(pred, label)
    recorder.train_step(0.15, 5e-4)
    recorder.valid_step(0.1)
    recorder.save_process()

    print(json.dumps(recorder.historical_metrics, ensure_ascii=False, indent=2))
