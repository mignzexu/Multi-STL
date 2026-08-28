import torch
from torch import nn
from .utils import (reshape_patch, reshape_patch_back)

try:
    from ..Model_system import distribute_model_layers
except ImportError:
    from models.Model_system import distribute_model_layers


class Main(nn.Module):

    def __init__(self, configs, device):
        super(Main, self).__init__()
        self.configs = configs
        self.device = device
        self.label_idx = configs["label_idx"]
        self.model = self.get_model()
        self.pre_seq_length = self.configs["total_seq"][0]
        self.aft_seq_length = self.configs["total_seq"][1]
        self.img_channel = len(self.configs["in_category"])
        self.img_height = self.configs["img_size"][0]
        self.img_width = self.configs["img_size"][1]

    def get_model(self):
        num_hidden = [int(x) for x in self.configs["num_hidden"].split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(num_layers, num_hidden, self.configs)

    def forward(self, batch_x, batch_y, **kwargs):
        # reverse schedule sampling
        if self.configs["reverse_scheduled_sampling"] == 1:
            mask_input = 1
        else:
            mask_input = self.pre_seq_length
        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.configs["patch_size"])

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            sum(self.configs["total_seq"]) - mask_input - 1,
            self.img_height // self.configs["patch_size"],
            self.img_width // self.configs["patch_size"],
            self.configs["patch_size"] ** 2 * self.img_channel)).to(batch_x.device)
            
        if self.configs["reverse_scheduled_sampling"] == 1:
            real_input_flag[:, :self.pre_seq_length - 1, :, :] = 1.0

        img_gen = self.model(test_dat, real_input_flag, return_loss=False)
        return  img_gen['pred']
    

class ConvLSTM_Model(nn.Module):

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()           
        C = len(configs["in_category"])
        H, W = configs["img_size"][0], configs["img_size"][1]
        self.configs = configs
        self.label_idx = configs["label_idx"]
        self.pre_seq_length = configs["total_seq"][0]
        self.aft_seq_length = configs["total_seq"][1]
        self.frame_channel = configs["patch_size"] * configs["patch_size"] * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs["patch_size"]
        width = W // configs["patch_size"]
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs["filter_size"],
                                       configs["stride"], configs["layer_norm"])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self._layer_devices = None

    def _get_layer_groups(self, devices):
        main_dev, hid_dev = devices[0], devices[1]
        return [
            (self.cell_list, hid_dev),
            (self.conv_last, main_dev),
        ]

    def forward(self, frames_tensor, mask_true, **kwargs):
        distribute_model_layers(self, self.configs, frames_tensor.device)
        main_dev = self._layer_devices[0]
        hid_dev = self._layer_devices[1]

        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to(hid_dev, non_blocking=True)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous().to(hid_dev, non_blocking=True)

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=hid_dev)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.pre_seq_length + self.aft_seq_length - 1):
            if self.configs["reverse_scheduled_sampling"] == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.pre_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1].to(main_dev, non_blocking=True))
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.configs["patch_size"])
        pred_y = next_frames[:, -self.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        return {'pred': pred_y}

    

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new