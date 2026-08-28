import torch
from torch import nn
import math
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
        self.patch_size = configs["patch_size"]
        self.pre_seq_length = self.configs["total_seq"][0]
        self.aft_seq_length = self.configs["total_seq"][1]
        self.total_length = sum(self.configs["total_seq"])
        self.img_channel = len(self.configs["in_category"])
        self.img_height = self.configs["img_size"][0]
        self.img_width = self.configs["img_size"][1]

    def get_model(self):
        num_hidden = [int(x) for x in self.configs["num_hidden"].split(',')]
        num_layers = len(num_hidden)
        return MAU_Model(num_layers, num_hidden, self.configs)

    def forward(self, batch_x, batch_y, **kwargs):
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.total_length - self.pre_seq_length - 1,
            self.img_height // self.patch_size,
            self.img_width // self.patch_size,
            self.patch_size ** 2 * self.img_channel)).to(batch_x.device)
        img_gen = self.model(test_ims, real_input_flag, return_loss=False)
        return img_gen["pred"]
    

class MAU_Model(nn.Module):
    
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MAU_Model, self).__init__()
        C = len(configs["in_category"])
        H, W = configs["img_size"][0], configs["img_size"][1]
        
        self.configs = configs
        self.label_idx = configs["label_idx"]
        self.pre_seq_length = configs["total_seq"][0]
        self.aft_seq_length = configs["total_seq"][1]
        self.total_length = sum(configs["total_seq"])
        self.frame_channel = configs["patch_size"] * configs["patch_size"] * C
        self.patch_size = configs["patch_size"]
        self.sr_size = configs["sr_size"]
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs["tau"]
        self.cell_mode = configs["cell_mode"]
        self.model_mode = configs["model_mode"]
        self.states = ['recall', 'normal']
        if not self.configs["model_mode"] in self.states:
            raise AssertionError
        cell_list = []

        width = W // self.patch_size // self.sr_size
        height = H // self.patch_size // self.sr_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs["filter_size"],
                        configs["stride"], self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(self.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)
        self._layer_devices = None

    def _get_layer_groups(self, devices):
        main_dev, hid_dev = devices[0], devices[1]
        return [
            (self.cell_list, hid_dev),
            (self.encoders, main_dev),
            (self.decoders, main_dev),
            (self.srcnn, main_dev),
            (self.merge, main_dev),
            (self.conv_last_sr, main_dev),
        ]

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        distribute_model_layers(self, self.configs, frames_tensor.device)
        main_dev = self._layer_devices[0]
        hid_dev = self._layer_devices[1]

        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to(main_dev, non_blocking=True)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous().to(main_dev, non_blocking=True)

        batch_size = frames.shape[0]
        height = frames.shape[3] // self.sr_size
        width = frames.shape[4] // self.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width], device=hid_dev))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width], device=hid_dev))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.total_length - 1):
            if t < self.pre_seq_length:
                net = frames[:, t]
            else:
                time_diff = t - self.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], height, width], device=hid_dev)
                    T_t.append(zeros)
            S_t = frames_feature.to(hid_dev, non_blocking=True)
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t.to(main_dev, non_blocking=True)

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)
        
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        next_frames = reshape_patch_back(next_frames, self.configs["patch_size"])
        pred_y = next_frames[:, -self.aft_seq_length:].contiguous()[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        return {'pred': pred_y}

class MAUCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        # self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.padding = filter_size // 2
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)

        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend
        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s

        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        return T_new, S_new
