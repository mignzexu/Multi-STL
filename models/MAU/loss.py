import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (reshape_patch, reserve_schedule_sampling_exp, schedule_sampling)

class loss_fn(nn.Module):

    def __init__(self, configs):
        super(loss_fn, self).__init__()
        self.configs = configs
        self.label_idx = configs['label_idx']
        self.global_step = 0
        self.eta = 1.0
        self.pre_seq_length = self.configs['total_seq'][0]
        self.img_channel = len(self.configs['in_category'])
        self.img_height = self.configs['img_size'][0]
        self.img_width = self.configs['img_size'][1]
        self.patch_size = self.configs['patch_size']
        self.loss_function = MSE_Loss()


    def forward(self, model, input, label, epoch = None):

        ims = torch.cat([input, label], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        ims = reshape_patch(ims, self.patch_size)

        eta, real_input_flag = schedule_sampling(self.eta, self.global_step, ims.shape[0], self.configs, input.device)

        real_input_flag = torch.zeros(
            (input.shape[0],
            sum(self.configs["total_seq"]) - self.pre_seq_length - 1,
            self.img_height // self.patch_size,
            self.img_width // self.patch_size,
            self.patch_size ** 2 * self.img_channel)).to(ims.device)
    
        output = model.model(ims, real_input_flag)
        label = label[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.loss_function(output, label)
        self.global_step += 1
        return loss, output


class MSE_Loss(nn.Module):

    def __init__(self):
        super(MSE_Loss, self).__init__()

        self.loss = nn.MSELoss()

    def forward (self, output, label):
        return self.loss(output["pred"], label)
    