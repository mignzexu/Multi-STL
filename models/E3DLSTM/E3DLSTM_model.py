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
        return E3DLSTM_Model(num_layers, num_hidden, self.configs)

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
    

class E3DLSTM_Model(nn.Module):
    r"""E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(E3DLSTM_Model, self).__init__()
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

        self.window_length = 2
        self.window_stride = 1

        height = H // configs["patch_size"]
        width = W // configs["patch_size"]
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channel, num_hidden[i],
                                  self.window_length, height, width, (2, 5, 5),
                                  configs["stride"], configs["layer_norm"]))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=(self.window_length, 1, 1),
                                   stride=(self.window_length, 1, 1), padding=0, bias=False)
        self._layer_devices = None

    def _get_layer_groups(self, devices):
        main_dev, hid_dev = devices[0], devices[1]
        return [
            (self.cell_list, hid_dev),
            (self.conv_last, main_dev),
        ]

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        distribute_model_layers(self, self.configs, frames_tensor.device)
        layer_devices = self._layer_devices
        assert layer_devices is not None
        main_dev = layer_devices[0]
        hid_dev = layer_devices[1]

        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to(hid_dev, non_blocking=True)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous().to(hid_dev, non_blocking=True)

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(
                torch.zeros_like(frames[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], self.window_length, height, width], device=hid_dev)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], self.window_length, height, width], device=hid_dev)

        for t in range(self.pre_seq_length + self.aft_seq_length - 1):
            # reverse schedule sampling
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

            input_list.append(net)

            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0)
                net = net.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                
                input = net if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1].to(main_dev, non_blocking=True)).squeeze(2)
            x_gen = x_gen.to(hid_dev, non_blocking=True)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.configs["patch_size"])
        pred_y = next_frames[:, -self.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        return {'pred': pred_y}


class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length,
                 height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = (0, filter_size[1] // 2, filter_size[2] // 2) 
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, window_length, height, width])
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, _, width, height = in_query.shape
        query = in_query.reshape(batch, -1, num_channels)
        keys = in_keys.reshape(batch, -1, num_channels)
        values = in_values.reshape(batch, -1, num_channels)
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = \
            torch.split(x_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)

        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)

        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory



class PredRNN_Model(nn.Module):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNN_Model, self).__init__()
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

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width,
                                       configs["filter_size"], configs["stride"], configs["layer_norm"]))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width]).to(device)

        for t in range(self.pre_seq_length + self.aft_seq_length - 1):
            # reverse schedule sampling
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

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.configs["patch_size"])
        pred_y = next_frames[:, -self.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        return {'pred': pred_y}

    
class SpatioTemporalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

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

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
