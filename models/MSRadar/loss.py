import torch
import torch.nn as nn



class WindWeightedLoss(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.thresholds = (1.6, 8.0, 17.2, 24.5)
        self.boost_factors = (1.0, 2.0, 5.0, 10.0)
        self.zero_event_weight = 0.3
        self.max_weight = 10.0
        self.normalization_scale = 30.0
        self.mode = mode

    
    def weights(self, real_wind_speed: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(real_wind_speed)

        for threshold, factor in zip(self.thresholds, self.boost_factors):
            mask = torch.sigmoid((real_wind_speed - threshold) * 5.0)
            weights = weights + (factor - 1.0) * mask

        weights = torch.where(real_wind_speed < self.thresholds[0], weights * self.zero_event_weight, weights)
        return torch.clamp(weights, 1.0, self.max_weight)

    def forward(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        weights = self.weights(ground_truth)
        if self.mode == "l1":
            absolute_error = torch.abs(output - ground_truth) / self.normalization_scale
        elif self.mode == "l2":
            absolute_error = (output - ground_truth) ** 2 / self.normalization_scale
        return torch.mean(weights * absolute_error)
       
