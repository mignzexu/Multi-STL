import json
import os
from datetime import datetime

import torch


class Recorder(object):
    def __init__(self, configs):
        self.configs = configs
        self.threshold = self.configs.threshold
        self.category = self.configs.out_category
        self.metrics = list(self.configs.metrics)
        self.current_epoch = 0
        self.file = {}
        self.historical_metrics = {}
        self.test_metrics = {}

        if len(self.threshold) != len(self.category):
            raise ValueError("阈值种类与类别数量不一致")

        self._reset_epoch_accumulators()

    def _reset_epoch_accumulators(self):
        cate_num = len(self.category)
        self.scalar_sums = {
            "mae": [0.0 for _ in range(cate_num)],
            "mse": [0.0 for _ in range(cate_num)],
            "rmse": [0.0 for _ in range(cate_num)],
        }
        self.scalar_counts = {
            "mae": [0 for _ in range(cate_num)],
            "mse": [0 for _ in range(cate_num)],
            "rmse": [0 for _ in range(cate_num)],
        }
        self.cvm_counts = []
        for category_thresholds in self.threshold:
            self.cvm_counts.append(
                [
                    {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    for _ in category_thresholds
                ]
            )

    def register_epoch(self, epoch):
        self.historical_metrics[str(epoch)] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train": {},
            "valid": {},
        }
        self.current_epoch = epoch

    def train_step(self, loss, lr):
        self.historical_metrics[str(self.current_epoch)]["train"]["loss"] = float(loss)
        self.historical_metrics[str(self.current_epoch)]["train"]["lr"] = float(lr)

    def valid_step(self, loss):
        self.historical_metrics[str(self.current_epoch)]["valid"]["loss"] = float(loss)
        metrics = self.metrics_statistics()
        self.historical_metrics[str(self.current_epoch)]["valid"].update(metrics)

    def test_step(self):
        metrics = self.metrics_statistics()
        self.test_metrics.clear()
        self.test_metrics.update(metrics)
        self.save_result()

    def metrics_statistics(self):
        cal_metrics = {}
        requested_metrics = set(self.metrics)
        eps = 1e-8

        for c, category_name in enumerate(self.category):
            cal_metrics[category_name] = {}

            if "mae" in requested_metrics:
                count = self.scalar_counts["mae"][c]
                value = self.scalar_sums["mae"][c] / count if count else 0.0
                cal_metrics[category_name]["mae"] = float(value)

            if "mse" in requested_metrics:
                count = self.scalar_counts["mse"][c]
                value = self.scalar_sums["mse"][c] / count if count else 0.0
                cal_metrics[category_name]["mse"] = float(value)

            if "rmse" in requested_metrics:
                count = self.scalar_counts["rmse"][c]
                mean_square = self.scalar_sums["rmse"][c] / count if count else 0.0
                cal_metrics[category_name]["rmse"] = float(mean_square ** 0.5)

            if "cvm" in requested_metrics:
                csi = []
                pod = []
                far = []
                hss = []

                for counts in self.cvm_counts[c]:
                    tp = counts["tp"]
                    fp = counts["fp"]
                    fn = counts["fn"]
                    tn = counts["tn"]
                    csi.append(round(tp / (tp + fn + fp + eps), 6))
                    pod.append(round(tp / (tp + fn + eps), 6))
                    far.append(round(fp / (tp + fp + eps), 6))
                    numerator = 2 * (tp * tn - fp * fn)
                    denominator = (tp + fn) * (tn + fn) + (tp + fp) * (tn + fp) + eps
                    hss.append(round(numerator / denominator, 6))

                cal_metrics[category_name]["csi"] = csi
                cal_metrics[category_name]["pod"] = pod
                cal_metrics[category_name]["far"] = far
                cal_metrics[category_name]["hss"] = hss

        self._reset_epoch_accumulators()
        return cal_metrics

    def save_process(self):
        save_path = os.path.join(self.configs.obj_dir, "metrics.json")
        os.makedirs(self.configs.obj_dir, exist_ok=True)
        try:
            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(self.historical_metrics, handle, ensure_ascii=False, indent=4)
        except Exception as exc:
            print("指标保存失败\n")
            print(exc)

    def save_result(self):
        save_path = os.path.join(self.configs.obj_dir, "result.json")
        os.makedirs(self.configs.obj_dir, exist_ok=True)
        try:
            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(self.test_metrics, handle, ensure_ascii=False, indent=4)
        except Exception as exc:
            print("指标保存失败\n")
            print(exc)

    def load_process(self):
        save_path = os.path.join(self.configs.obj_dir, "metrics.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as handle:
                self.historical_metrics = json.load(handle)

    @torch.no_grad()
    def within_the_epoch(self, pred, label):
        requested_metrics = set(self.metrics)

        for i in range(len(self.category)):
            pred_channel = pred[:, :, i, :, :]
            label_channel = label[:, :, i, :, :]

            if "mae" in requested_metrics:
                abs_error = torch.abs(pred_channel - label_channel)
                self.scalar_sums["mae"][i] += float(abs_error.sum().item())
                self.scalar_counts["mae"][i] += int(abs_error.numel())

            if "mse" in requested_metrics or "rmse" in requested_metrics:
                square_error = (pred_channel - label_channel) ** 2
                if "mse" in requested_metrics:
                    self.scalar_sums["mse"][i] += float(square_error.sum().item())
                    self.scalar_counts["mse"][i] += int(square_error.numel())
                if "rmse" in requested_metrics:
                    self.scalar_sums["rmse"][i] += float(square_error.sum().item())
                    self.scalar_counts["rmse"][i] += int(square_error.numel())

            if "cvm" in requested_metrics:
                for threshold_index, threshold_value in enumerate(self.threshold[i]):
                    counts = self.cvm_counts[i][threshold_index]
                    counts["tp"] += int(torch.sum((pred_channel >= threshold_value) & (label_channel >= threshold_value)).item())
                    counts["fp"] += int(torch.sum((pred_channel >= threshold_value) & (label_channel < threshold_value)).item())
                    counts["fn"] += int(torch.sum((pred_channel < threshold_value) & (label_channel >= threshold_value)).item())
                    counts["tn"] += int(torch.sum((pred_channel < threshold_value) & (label_channel < threshold_value)).item())

    @staticmethod
    def mae(pred, true):
        return torch.mean(torch.abs(pred - true)).item()

    @staticmethod
    def mse(pred, true):
        return torch.mean((pred - true) ** 2).item()

    @staticmethod
    def rmse(pred, true):
        return torch.sqrt(torch.mean((pred - true) ** 2)).item()

    @staticmethod
    def cvm(pred, true, threshold):
        metrics = [[], [], [], []]
        for value in threshold:
            tp = torch.sum((pred >= value) & (true >= value)).item()
            fp = torch.sum((pred >= value) & (true < value)).item()
            fn = torch.sum((pred < value) & (true >= value)).item()
            tn = torch.sum((pred < value) & (true < value)).item()
            metrics[0].append(tp / (tp + fn + fp + 1e-8))
            metrics[1].append(tp / (tp + fn + 1e-8))
            metrics[2].append(fp / (tp + fp + 1e-8))
            denominator = (tp + fn) * (tn + fn) + (tp + fp) * (tn + fp) + 1e-8
            metrics[3].append((2 * (tp * tn - fp * fn)) / denominator)
        return metrics
