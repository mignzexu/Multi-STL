import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_fn(nn.Module):

    def __init__(self, configs):
        super(loss_fn, self).__init__()
        self.label_idx = configs['label_idx']
        self.loss_function = TAU_Loss(configs['alpha'])


    def forward(self, model, input, label, epoch = None):
        output = model.model(input)
        label = label[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.loss_function(output, label)
        return loss, output


class TAU_Loss(nn.Module):

    def __init__(self, alpha):
        super(TAU_Loss, self).__init__()

        self.loss = nn.MSELoss()
        self.alpha = alpha

    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()

    def forward (self, output, label):
        loss = self.loss(output["pred"], label) + self.alpha * self.diff_div_reg(output["pred"], label)
        return loss
    