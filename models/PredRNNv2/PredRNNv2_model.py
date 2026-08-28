import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
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
        return PredRNNv2_Model(num_layers, num_hidden, self.configs)

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
    

class PredRNNv2_Model(nn.Module):
    r"""PredRNNv2 Model

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2_Model, self).__init__()
        C = len(configs["in_category"])
        H, W = configs["img_size"][0], configs["img_size"][1]

        self.configs = configs
        self.label_idx = configs["label_idx"]
        self.pre_seq_length = configs["total_seq"][0]
        self.aft_seq_length = configs["total_seq"][1]
        self.total_length = sum(configs["total_seq"])
        self.frame_channel = configs["patch_size"] * configs["patch_size"] * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs["patch_size"]
        width = W // configs["patch_size"]

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], height, width,
                                         configs["filter_size"], configs["stride"], configs["layer_norm"]))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        self._layer_devices = None

    def _get_layer_groups(self, devices):
        main_dev, hid_dev = devices[0], devices[1]
        return [
            (self.cell_list, hid_dev),
            (self.adapter, hid_dev),
            (self.conv_last, main_dev),
        ]

    def forward(self, frames_tensor, mask_true, **kwargs):
        return_loss = kwargs.get('return_loss', True)
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
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
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width], device=hid_dev)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width], device=hid_dev)

        for t in range(self.total_length - 1):

            if self.configs["reverse_scheduled_sampling"] == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.pre_seq_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1].to(main_dev, non_blocking=True))
            next_frames.append(x_gen)

            # decoupling loss
            if return_loss:
                for i in range(0, self.num_layers):
                    decouple_loss.append(torch.mean(torch.abs(
                        torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if return_loss:
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.configs["patch_size"])
        pred_y = next_frames[:, -self.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()[:, :, self.label_idx[0]:self.label_idx[1], :, :]

        return {"pred": pred_y, "decouple_loss": decouple_loss}

class SpatioTemporalLSTMCellv2(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellv2, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
