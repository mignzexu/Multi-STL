import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_fn(nn.Module):

    def __init__(self, loss_name, configs):
        super(loss_fn, self).__init__()

        self.loss_name = loss_name
        self.label_idx = configs['label_idx']
        self.loss_function = self.load_loss()
    

    def load_loss(self):
        try:
            if self.loss_name == 'mse':
                loss_function = MSE_Loss()
            else:
                raise NotImplementedError(f'未找到损失函数{self.loss_name}')
            
            return loss_function

        except Exception as e:
            print(f'未找到模型损失函数:{e}')
            exit()

    def forward(self, model, input, label):
        output = model.model(input)
        label = label[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.loss_function(output, label)
        return loss, output



class MSE_Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward (self, output, label):
        return self.loss(output['pred'], label)