import torch
import torch.nn as nn
import numpy as np



class None_std:
    def __init__(self, configs):
        self.configs = configs

    def cal_params(self, data):
        pass

    def load_params(self, metric = False):
        pass

    def standardizing(self, data):
        return data

    def de_standardizing(self, data):
        return data


class Z_Score(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    @classmethod
    def numpy_mean_std(cls, data, chunk_size=512, eps=1e-6):
        """
        使用 NumPy/memmap 分块计算 mean/std。

        输入:
            data: numpy.ndarray 或 numpy.memmap
            shape: [T, C, H, W]

        返回:
            data_mean: torch.Tensor, shape [1, C, 1, 1]
            data_std : torch.Tensor, shape [1, C, 1, 1]
        """

        if data.ndim != 4:
            raise ValueError(f"data 应该是 [T, C, H, W]，但得到 shape={data.shape}")

        total_t, channels, height, width = data.shape

        channel_sum = np.zeros(channels, dtype=np.float64)
        channel_sumsq = np.zeros(channels, dtype=np.float64)
        total_count = 0

        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)

            # 只读取当前时间片，不会完整加载 data
            chunk = data[start:end]
            chunk64 = np.asarray(chunk, dtype=np.float64)

            channel_sum += chunk64.sum(axis=(0, 2, 3))
            channel_sumsq += np.square(chunk64).sum(axis=(0, 2, 3))

            total_count += chunk64.shape[0] * height * width

            del chunk64

        mean = channel_sum / total_count
        var = channel_sumsq / total_count - np.square(mean)
        var = np.maximum(var, 0.0)

        std = np.sqrt(var)
        std = np.where(std < eps, 1.0, std)

        data_mean = torch.from_numpy(mean.astype(np.float32)).view(1, -1, 1, 1)
        data_std = torch.from_numpy(std.astype(np.float32)).view(1, -1, 1, 1)

        return data_mean, data_std

    def cal_params(self, data, label_idx):
        if isinstance(data, torch.Tensor):
            self.data_mean = torch.mean(data, dim=(0, 2, 3)).view(1, -1, 1, 1)
            self.data_std = torch.std(data, dim=(0, 2, 3)).view(1, -1, 1, 1)

        elif isinstance(data, (np.ndarray, np.memmap)):
            self.data_mean, self.data_std = self.numpy_mean_std(
                data,
                chunk_size=getattr(self.configs, "std_chunk_size", 512),
                eps=getattr(self.configs, "std_eps", 1e-6),
            )

        else:
            raise TypeError(
                f"不支持的数据类型: {type(data)}，"
                f"期望 torch.Tensor、np.ndarray 或 np.memmap"
            )

        self.metric_mean = self.data_mean[:, label_idx[0]:label_idx[1], :, :].unsqueeze(0)
        self.metric_std = self.data_std[:, label_idx[0]:label_idx[1], :, :].unsqueeze(0)

        self.configs.std_params = {
            "dataset": {
                "mean": self.data_mean.tolist(),
                "std": self.data_std.tolist(),
            },
            "metric": {
                "mean": self.metric_mean.tolist(),
                "std": self.metric_std.tolist(),
            },
        }

    def data_params(self):
        """
        在 datasets 和 test 保存结果时被调用。
        """
        self.data_mean = torch.tensor(self.configs.std_params["dataset"]["mean"])
        self.data_std = torch.tensor(self.configs.std_params["dataset"]["std"])

    def metric_params(self):
        """
        在模型训练和测试指标中被调用。
        """
        self.register_buffer(
            "metric_mean",
            torch.tensor(self.configs.std_params["metric"]["mean"])
        )
        self.register_buffer(
            "metric_std",
            torch.tensor(self.configs.std_params["metric"]["std"])
        )

    def standardizing(self, data):
        return (data - self.data_mean) / self.data_std

    def de_standardizing(self, data, op="metric"):
        if op == "metric":
            return data * self.metric_std + self.metric_mean
        else:
            return data * self.data_std + self.data_mean


class Z_Score_SD(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.in_category = self.configs.in_category
        self.no_norm_category = ["LT"]


    def cal_params(self, data, label_idx):
        self.data_mean = torch.mean(data, dim=(0, 2, 3)).view(1, -1, 1, 1)
        self.data_std = torch.std(data, dim=(0, 2, 3)).view(1, -1, 1, 1)
        self.data_mean, self.data_std = self.disable_norm(self.data_mean, self.data_std)
        self.metric_mean = self.data_mean[:, label_idx[0]:label_idx[1], :, :].unsqueeze(0)
        self.metric_std = self.data_std[:, label_idx[0]:label_idx[1], :, :].unsqueeze(0)
        self.configs.std_params = {
            "dataset" : {
                "mean": self.data_mean.tolist(),
                "std": self.data_std.tolist()
            },
            "metric" : {
                "mean": self.metric_mean.tolist(),
                "std": self.metric_std.tolist()
            }
        }

    def disable_norm(self, mean, std):
        """
        将指定类别通道改成不归一化:
        mean -> 0
        std  -> 1
        """
        mean = mean.clone()
        std = std.clone()

        for i, cat in enumerate(self.in_category):
            if cat in self.no_norm_category:
                mean[..., i, :, :] = 0.0
                std[..., i, :, :] = 1.0
        
        return mean, std


    def data_params(self):
        """
           在datasets和test保存结果时被调用
        """
        self.data_mean = torch.tensor(self.configs.std_params["dataset"]["mean"])
        self.data_std = torch.tensor(self.configs.std_params["dataset"]["std"])

    def metric_params(self):
        """
           在模型训练和测试指标中被调用
        """
        self.register_buffer("metric_mean", torch.tensor(self.configs.std_params["metric"]["mean"]))
        self.register_buffer("metric_std", torch.tensor(self.configs.std_params["metric"]["std"]))

    def standardizing(self, data):
        return (data - self.data_mean) / self.data_std

    def de_standardizing(self, data, op = "metric"):
        if op == "metric":
            return data * self.metric_std + self.metric_mean
        else:
            return data * self.data_std + self.data_mean
        






class Z_Score_2(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.threshold = {
            "w10": 30.0,
            "CW" : 30.0,
            "tp": 1.0,
            "terrain": 2000.0
        }


    def cal_params(self, data, label_idx):
        params = []
        for cate in self.in_category:
            params.append(self.threshold[cate])

        params = torch.tensor(params).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.data_p = params
        self.metric_p = params[:, label_idx[0]:label_idx[1], :, :].unsqueeze(0)
        self.configs.std_params = {
            "dataset" : {
                "mean": self.data_p.tolist()
            },
            "metric" : {
                "mean": self.metric_p.tolist()
            }
        }

    def data_params(self):
        """
           在datasets和test保存结果时被调用
        """
        self.data_mean = torch.tensor(self.configs.std_params["dataset"]["mean"])
        self.data_std = torch.tensor(self.configs.std_params["dataset"]["std"])

    def metric_params(self):
        """
           在模型训练和测试指标中被调用
        """ 
        self.register_buffer("metric_mean", torch.tensor(self.configs.std_params["metric"]["mean"]))
        self.register_buffer("metric_std", torch.tensor(self.configs.std_params["metric"]["std"]))

    def standardizing(self, data):
        return (data - self.data_mean) / self.data_std

    def de_standardizing(self, data, op = "metric"):
        if op == "metric":
            return data * self.metric_std + self.metric_mean
        else:
            return data * self.data_std + self.data_mean
