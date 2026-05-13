import torch
import torch.nn as nn


class loss_fn(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.label_idx = configs["label_idx"]
        self.out_channels = len(configs["out_category"])
        self.metric_sampling_steps = int(configs.get("prediff_metric_sampling_steps", 8))

    def _slice_label(self, label):
        start, end = self.label_idx
        if label.dim() == 5 and label.shape[2] != self.out_channels:
            return label[:, :, start:end, :, :]
        if label.dim() == 4 and label.shape[1] != self.out_channels:
            return label[:, start:end, :, :]
        return label

    def forward(self, model, input, label, epoch=None):
        target = self._slice_label(label)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        loss = model.model.training_loss(model._prepare_input_seq(input), target)

        if model.training:
            pred = model.model.persistence_predict(model._prepare_input_seq(input), steps=target.shape[1])
        else:
            with torch.no_grad():
                pred = model.predict(input, steps=target.shape[1], sample_steps=self.metric_sampling_steps)

        output = {
            "pred": pred,
            "metric_pred": pred,
            "metric_target": target,
        }
        return loss, output
