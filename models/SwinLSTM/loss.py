import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_fn(nn.Module):

    def __init__(self, loss_name, configs):
        super(loss_fn, self).__init__()

        self.loss_name = loss_name
        self.label_idx = configs['label_idx']
        self.loss_function = MSE_Loss()
    
    def forward(self, model, input, label, epoch = None):
        ims = torch.cat([input, label], dim=1).permute(0, 1, 3, 4, 2).contiguous() # [8, 20, 64, 64, 1] for mmnist
        output = model.model(ims)
        label = label[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.loss_function(output, label)
        return loss, output


class MSE_Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward (self, output, label):
        return self.loss(output['pred'], label)