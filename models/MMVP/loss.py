import torch.nn as nn
import torch.nn.functional as F

class loss_fn(nn.Module):

    def __init__(self, configs):
        super(loss_fn, self).__init__()
        self.label_idx = configs['label_idx']
        self.loss_function = MSE_Loss()

    def forward(self, model, input, label, epoch = None):
        output = model.model(input)
        label = label[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.loss_function(output, label)
        return loss, output


class MSE_Loss(nn.Module):

    def __init__(self):
        super(MSE_Loss, self).__init__()

        self.loss = nn.MSELoss()

    def forward (self, output, label):
        return self.loss(output["pred"], label)
    