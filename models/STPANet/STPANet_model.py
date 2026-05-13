import torch
import math 
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class Main(nn.Module):

    def __init__(self, configs, device):
        super(Main, self).__init__()
        self.configs = configs
        self.device = device
        self.model = STPANet_Model(self.configs)
        self.pre_seq_length = self.configs["total_seq"][0]
        self.aft_seq_length = self.configs["total_seq"][1]

    def forward(self, batch_x, batch_y=None):
        if self.aft_seq_length == self.pre_seq_length:
            pred_y = self.model(batch_x)["pred"]
        elif self.aft_seq_length < self.pre_seq_length:
            pred_y = self.model(batch_x)["pred"]
            pred_y = pred_y[:, : self.aft_seq_length]
        elif self.aft_seq_length > self.pre_seq_length:
            pred_y = []
            d = self.aft_seq_length // self.pre_seq_length
            m = self.aft_seq_length % self.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)["pred"]
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)["pred"]
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)

        return pred_y

class STPANet_Model(nn.Module):

    def __init__(self, configs):
        super(STPANet_Model, self).__init__()

        self.T = configs["total_seq"][0]
        self.C_in = len(configs["in_category"])
        self.C_out = len(configs["out_category"])
        H, W = configs["img_size"][0], configs["img_size"][1]
        N_S = configs["N_S"]
        N_T = configs["N_T"]
        hid_S = configs["hid_S"]
        hid_T = configs["hid_T"]
        mlp_ratio = configs["mlp_ratio"]
        drop = configs["drop"]
        drop_path = configs["drop_path"]
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # 下采样 1 / 2**(N_S/2)
        act_inplace = False # 如果是 inplace，会导致梯度爆炸
        self.enc = Encoder(self.C_in, hid_S, N_S, configs["spatio_kernel_enc"], act_inplace=act_inplace)
        self.dec = Decoder(hid_S, self.C_out, N_S, configs["spatio_kernel_dec"], act_inplace=act_inplace)
        self.hid = MidMetaNet(self.T*hid_S, hid_T, N_T,
                input_resolution=(H, W),
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)


    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape     #[1, 12, 1, 32, 64]
        x = x_raw.view(B*T, C, H, W)     #[12, 1, 32, 64]

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape # [12, 32, 16, 32] 经过encoder后,元数据的通道数变为32,长宽分辨率变小为原来的一半

        z = embed.view(B, T, C_, H_, W_)    # [B, 12, 32, 16, 32]
        hid = self.hid(z)   
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid ,skip)
        Y = Y.reshape(B, T, self.C_out, H, W)
        return {"pred": Y}
    

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)   # [12, 32, 32, 64]
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        # Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y
    

class MidMetaNet(nn.Module):


    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y
    

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, 
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = STPASubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x) #[1, 384, 16, 32]convmixer
        return z if self.in_channels == self.out_channels else self.reduction(z)
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class STPABlock(nn.Module):
   
    def __init__(self, dim, pool_size=3, mlp_ratio=4., drop=0., drop_path=0.,
                 init_value=1e-5, 
                #  act_layer=nn.SiLU, 
                 act_layer=nn.GELU,
                 norm_layer=GroupNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.channel_pooling = Channel_Pooling(dim)
        self.msse = MSSEModule_2(dim)
        self.token_mixer = Pooling_4(dim, pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = STMLP(in_features=dim, mlp_redio=mlp_ratio, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):

        m = self.msse(self.norm1(x))
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(m) * self.channel_pooling(m))

        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
    
class Channel_Pooling(nn.Module):   
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim , dim, bias=False), # expansion
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten

class STPASubBlock(STPABlock):
    """A block of STPA."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path,
                         drop=drop, init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MSSEModule_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.Dconv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", groups=dim, dilation=1)
        self.Dconv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", groups=dim, dilation=2)   
        self.Dconv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", groups=dim, dilation=3)   
        self.cat_conv1 = nn.Conv2d(dim * 3, dim * 16, kernel_size=1, stride=1, padding=0, groups=dim)
        self.cat_conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, groups=dim)

    def forward(self, x):
        
        x1 = self.Dconv_1(x)
        x2 = self.Dconv_2(x)
        x3 = self.Dconv_3(x)
        m = torch.stack([x1, x2, x3], dim=2)  # [B, C, 3, H, W]
        m = m.flatten(1,2)
        m = F.gelu(self.cat_conv1(m))  # [B, C*16, H, W]
        m = m.unflatten(1, (self.dim, 16))  # [B, C, 16, H, W]
        max_pool = torch.max(m, dim=2).values  # [B, C, H, W]
        avg_pool = torch.mean(m, dim=2) # [B, C, H, W]
        out = torch.stack([max_pool, avg_pool], dim=2)  # [B, C, 2, H, W]
        out = out.flatten(1,2)  # [B, C*2, H, W]
        out = F.sigmoid(self.cat_conv2(out))  # [B, C, H, W]
        x = x * out  # [B, C, H, W]
        return x


class Pooling_4(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size +o
    """
    def __init__(self, dim, pool_size=3):
        super().__init__()
        self.conv0 = nn.Conv2d(
            3 * dim, 16 * dim, kernel_size=1, stride=1, padding=0, groups=dim)
        self.conv1 = nn.Conv2d(
            16 * dim, dim, kernel_size=1, stride=1, padding=0, groups=dim)
        self.avg_pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=True)
        self.max_pool = nn.MaxPool2d(
            pool_size, stride=1, padding=pool_size//2)
        self.min_pool = MinPool2d(
            pool_size, stride=1, padding=pool_size//2)
        self.instnorm_1 = nn.InstanceNorm2d(16*dim, affine=True)
        self.instnorm_2 = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        min_x = self.min_pool(x)
        # gate = torch.stack([max_x, min_x], dim=2)  # [B, C, 3, H, W]
        # gate = gate.mean(2)  # [B, C, H, W]
        # gate = F.sigmoid(self.instnorm_2(avg_x - gate))  # [B, C, H, W]
        att = torch.stack([max_x, avg_x, min_x], dim=2)  # [B, C, 3, H, W]
        att = att.flatten(1,2)  # [B, C*3, H, W]
        att = F.gelu(self.instnorm_1(self.conv0(att)))   # [B, C*3, H, W]
        # att = self.conv1(att) * gate   # [B, C, H, W]
        att = self.conv1(att)
        result = att - x
        return result
    
class MinPool2d(nn.Module):
    """
    最小池化操作的实现
    通过对输入取负，使用MaxPool2d，然后再取负来实现
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.maxpool(-x)
    

class STMLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features,  mlp_redio, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.dwconv_up = nn.Conv2d(in_features, hidden_features, kernel_size=5, bias=False, groups=in_features, padding="same")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_redio = int(mlp_redio)
        self.in_features = in_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        mlp = self.fc1(x)
        dw_up = self.dwconv_up(x)
        mlp = self.act(mlp)
        dw_up = dw_up * F.sigmoid(self.avg_pool(mlp))

        dw_up = dw_up.unflatten(1, (self.in_features, self.mlp_redio))  #BCDHW
        dw_up = F.sigmoid(torch.mean(dw_up, dim=2)) # [B, C, H, W]
        mlp = self.fc2(mlp)
        mlp = mlp * dw_up

        return mlp