import torch.nn as nn


class loss_fn(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs

        self.label_idx = self.configs.label_idx
        self.loss_function = MSE_Loss()

    def forward(self, model, input_ims, label_ims):
        pred = model.model(input_ims)
        label = label_ims[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        output = pred if isinstance(pred, dict) else {"pred": pred}
        loss = self.loss_function(output, label)
        return loss, output


class MSE_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, label):
        return self.loss(output["pred"], label)
