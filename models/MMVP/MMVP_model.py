import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Main(nn.Module):

    def __init__(self, configs, device):
        super(Main, self).__init__()
        self.configs = configs
        self.device = device
        self.model = MMVP_Model(self.configs)
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
    
class MMVP_Model(nn.Module):
    def __init__(self, configs):
        super(MMVP_Model, self).__init__()

        T = configs["total_seq"][0]
        C = len(configs["in_category"])
        H, W = configs["img_size"][0], configs["img_size"][1]
        downsample_ratio = [int(x) for x in configs["downsample_setting"].split(',')]
        highres_scale = np.prod(downsample_ratio[:-1]) * 2
        lowres_scale = np.prod(downsample_ratio) * 2
        hid_S = configs["hid_S"]
        hid_T = configs["hid_T"]
        rrdb_encoder_num = configs["rrdb_encoder_num"]
        rrdb_decoder_num = configs["rrdb_decoder_num"]
        aft_seq_length = configs["total_seq"][1]
        self.pre_seq_length = T
        self.mat_size = [[H // highres_scale, W // highres_scale], [H // lowres_scale, W // lowres_scale]]
        
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)
        self.enc = RRDBEncoder(C=C, hid_S=hid_S, rrdb_encoder_num=rrdb_encoder_num, downsample_ratio=downsample_ratio)
        self.filter = filter_block(downsample_scale=downsample_ratio, hid_S=hid_S, mat_size=self.mat_size)
        self.dec = RRDBDecoder(C=C, hid_S=hid_S, rrdb_decoder_num=rrdb_decoder_num, downsample_scale=downsample_ratio)
        self.fuse = Compose(downsample_scale=downsample_ratio, mat_size=self.mat_size, 
                            prev_len=T, aft_seq_length=aft_seq_length)

        self.hid = MidMotionMatrix(T=T, hid_S=hid_S, hid_T=hid_T, mat_size=self.mat_size, 
                                    aft_seq_length=aft_seq_length, use_direct_predictor=configs["use_direct_predictor"])

        res_shuffle_scale = 1
        for s in range(len(downsample_ratio) - 1):
            res_shuffle_scale *= downsample_ratio[s]
        self.res_shuffle = nn.PixelShuffle(res_shuffle_scale)
        self.res_unshuffle = nn.PixelUnshuffle(res_shuffle_scale)
        self.res_shuffle_scale = res_shuffle_scale

        self.enhance = ImageEnhancer(C_in=C, hid_S=hid_S, downsample_scale=downsample_ratio, rrdb_enhance_num=configs["rrdb_enhance_num"])

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x_raw = x_raw.reshape(B*T, C, H, W)
        x_raw = self.unshuffle(x_raw)
        x = x_raw.clone()
        x_wh = x.shape[-2:]
        # encoder
        fi = self.enc(x)  #N, C, H, W

        #record fi shape for later decoder
        feat_shape = []    
        for i in range(len(fi)):
            if fi[i] is None:
                feat_shape.append(None)
            else:
                feat_shape.append(fi[i].shape[2:])
        #filter block
        gi = self.filter(fi)

        #construct and predict similarity matrix
        similarity_matrix = self.hid(gi, B, T)

        # compose motion matrix and embed feature
        composed_fut_feat = self.fuse(fi, similarity_matrix, feat_shape)

        #decoder feature
        recon_img = self.dec(composed_fut_feat)
        final_recon_img = recon_img.clone()

        if x_wh != recon_img.shape[2:]:
            std_w = int(self.mat_size[0][0] * self.res_shuffle_scale)
            std_h = int(self.mat_size[0][1] * self.res_shuffle_scale)
            x_raw = F.interpolate(x_raw, (std_w, std_h))

        image_list = [self.res_unshuffle(x_raw)]
        compose_image, avg_image = self.fuse.feat_compose(image_list, [similarity_matrix[0]])
        compose_image = compose_image[0]
        
        compose_image = self.res_shuffle(compose_image)
        fut_img_seq = self.shuffle(compose_image)

        recon_img = self.shuffle(recon_img)
        final_recon_img = self.shuffle(final_recon_img)
        if fut_img_seq.shape[2:] != final_recon_img.shape[2:]:
            fut_img_seq = F.interpolate(fut_img_seq, final_recon_img.shape[2:])
        final_recon_img = self.enhance(torch.cat([final_recon_img, fut_img_seq], dim=1))

        if recon_img.shape[-2] != H or recon_img.shape[-1] != W:
            recon_img = F.interpolate(recon_img, (H, W))
            final_recon_img = F.interpolate(final_recon_img, (H, W))
        
        recon_img = recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)
        final_recon_img = final_recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)

        return {"pred": final_recon_img}
        
        
        
class RRDBEncoder(nn.Module):
    def __init__(self, C=1, hid_S=32, downsample_ratio=[2, 2, 2], rrdb_encoder_num=2, scale_in_use=3):
        super(RRDBEncoder, self).__init__()
        self.C_in = C * 4
        self.hid_S = hid_S
        self.scale_num = len(downsample_ratio)
        self.downsample_ratio = downsample_ratio
        self.scale_in_use = scale_in_use
        self.inconv = nn.Conv2d(self.C_in, self.hid_S, 3, 1, 1)
        self.block_rrdb = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_encoder_num)])

        pre_downsample_block_list = []
        
        for i in range(self.scale_num-2):
            pre_downsample_block_list.append(ResBlock(hid_S * (2 ** i), hid_S * (2 ** (i+1)), 
                                                      downsample=True, factor=downsample_ratio[i]))
        self.pre_downsample_block = nn.ModuleList(pre_downsample_block_list)
        
        self.downsample_high = ResBlock(hid_S * ( 2 ** (self.scale_num-2)), hid_S * ( 2 ** (self.scale_num-1)),
                                        downsample=True, factor=downsample_ratio[-2])
        self.downsample_low = ResBlock(hid_S * (2 ** (self.scale_num-1)), hid_S * (2 ** (self.scale_num)),
                                       downsample=True, factor=downsample_ratio[-1])


    def forward(self, x, save_all=False):
        in_feat = []
        x = self.inconv(x)
        x = self.block_rrdb(x)
        in_feat.append(x)
        for i in range(self.scale_num-2):
            x = self.pre_downsample_block[i](x) 
            in_feat.append(x)
        x = self.downsample_high(x)
        in_feat.append(x)
        x = self.downsample_low(x)
        in_feat.append(x)
        if self.scale_in_use == 3:
            for i in range(len(in_feat) - 3):
                in_feat[i] = None
        elif self.scale_in_use == 2:
            for i in range(len(in_feat)-2):
                in_feat[i] = None
        return in_feat

class filter_block(nn.Module):
    def __init__(self, downsample_scale, hid_S, mat_size):
        super(filter_block, self).__init__()
        self.filter_block = []
        high_scale = len(downsample_scale) - 1
        feat_len = hid_S * (2 ** high_scale)
        self.mat_size = mat_size
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU()))
        low_scale = high_scale + 1
        feat_len = hid_S * (2 ** low_scale)
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU()))
    
        self.filter_block = nn.ModuleList(self.filter_block)

    def forward(self, x):
        gi = []
        for s in [-2, -1]:
            feat_area = x[s].shape[-1] * x[s].shape[-2]
            mat_area = self.mat_size[s][-1] * self.mat_size[s][-2]
            if mat_area != feat_area:
                out = F.interpolate(x[s].clone(), size=tuple(self.mat_size[s]), mode='bilinear')
            else:
                out = x[s].clone()
            out = self.filter_block[s](out)
            gi.append(out)
        return gi

        

class MidMotionMatrix(nn.Module):
    def __init__(self, T, hid_S=32, hid_T=192, mat_size=[[8, 8], [4, 4]], 
                 aft_seq_length=10, use_direct_predictor=True):
        super(MidMotionMatrix, self).__init__()
        self.pre_seq_len = T
        self.mat_size = mat_size
        self.mx_h = mat_size[0][0]
        self.mx_w = mat_size[0][1]
        self.scale_fuser_1 = Up(hid_S * 2, hid_S, bilinear=False, scale=2)
        self.scale_fuser_2 = nn.Sequential(nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(hid_S),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(hid_S),
                                           nn.LeakyReLU())
        self.predictor = PredictModel(T=T, hidden_len=hid_T, aft_seq_length=aft_seq_length,
                                      mx_h=self.mx_h, mx_w=self.mx_w,
                                      use_direct_predictor=use_direct_predictor)

    def forward(self, x, B, T):
        similar_matrix = []
        prev_sim_matrix = []
        pred_sim_matrix = [None, None]
        # construct similarity matrix
        for i in [-2, -1]:
            N = x[i].shape[0]
            h, w = x[i].shape[2:]
            cur_sim_matrix = build_similarity_matrix(x[i].reshape(B, T, -1, h, w))
            prev_sim_matrix.append(cur_sim_matrix[:, :self.pre_seq_len-1].clone())
        
        pred_fut_matrix, _ = self.predictor(prev_sim_matrix[0], softmax=False, res=None)
        pred_sim_matrix[0] = pred_fut_matrix.clone()
        pred_sim_matrix[1] = sim_matrix_interpolate(pred_fut_matrix.clone(), self.mat_size[0], self.mat_size[1])
        # post process the matrix
        pred_sim_matrix[0] = sim_matrix_postprocess(pred_sim_matrix[0])
        pred_sim_matrix[1] = sim_matrix_postprocess(pred_sim_matrix[1])

        #update similarity matrix list
        for i in range(len(prev_sim_matrix)):
            new_cur_sim_matrix = torch.cat([sim_matrix_postprocess(prev_sim_matrix[i]), pred_sim_matrix[i]], dim=1)
            similar_matrix.append(new_cur_sim_matrix)
        return similar_matrix

class Compose(nn.Module):
    def __init__(self, downsample_scale, mat_size, prev_len, aft_seq_length):
        super(Compose, self).__init__()
        self.downsample_scale = downsample_scale
        self.mat_size = mat_size
        self.prev_len = prev_len
        self.aft_seq_length = aft_seq_length
        self.feat_shuffle = []
        self.feat_unshuffle = []
        self.feat_scale_list = []
        for i in range(len(self.downsample_scale) - 1):
            feat_shuffle_scale = 1
            for s in range(len(self.downsample_scale) - 2, i - 1, -1):
                feat_shuffle_scale *= self.downsample_scale[s]
            self.feat_scale_list.append(feat_shuffle_scale)
            self.feat_shuffle.append(nn.PixelShuffle(feat_shuffle_scale))
            self.feat_unshuffle.append(nn.PixelUnshuffle(feat_shuffle_scale))
        self.feat_shuffle = nn.ModuleList(self.feat_shuffle)
        self.feat_unshuffle = nn.ModuleList(self.feat_unshuffle)

    def feat_generator(self, feats, sim_matrix, feat_idx, img_compose=False, scale=1):
        '''

        :param feats: [B,T,c,h,w]
        :param sim_matrix: [B,T,h*w,h*w]
        :return: new_feats: [B,c,h,w]
        '''
        B, T, c, h, w = feats.shape
        # only test single motion
        if scale > 1: # if hw_cur != hw_target, only use the last sim matrix
            feats = feats[:,-1:,]
            sim_matrix = sim_matrix[:,-1:]
            T = 1
        feats = feats.permute(0, 2, 1, 3, 4)  # (B,c,T,h,w)
        feats = feats.reshape(B, c, T * h * w).permute(0, 2, 1)  # (B,Prev T*h*w,c)
        B,T,hw_cur,hw_target = sim_matrix.shape
        sim_matrix = sim_matrix.reshape(B, T * hw_cur, hw_target).permute(0, 2, 1) # Batch, fut H*W, Prev T*HW
        weight = torch.sum(sim_matrix, dim=-1).reshape(-1, 1, hw_target) + 1e-6
        new_feats = torch.bmm(sim_matrix, feats).permute(0, 2, 1) / weight
        new_feats = new_feats.reshape(B, c, h*scale, w*scale)

        return new_feats

    def feat_compose(self, emb_feat_list, sim_matrix, img_compose=False, scale=1, use_gt=False):
        '''

        :param emb_feat_list: (scale_num, (B,T,c,h,w))
        :param sim_matrix:  (B,T-1,h,w,h,w)
        :param use_gt_sim_matrix: bool
        :return: fut_emb_feat_list (scale_num, (B,t,c,h,w))
        '''
        fut_emb_feat_list = []
        ori_emb_feat_list = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                fut_emb_feat_list.append(None)
                ori_emb_feat_list.append(None)
                continue

            fut_emb_feat_list.append([])
            cur_emb_feat = emb_feat_list[i]
            ori_emb_feat_list.append(torch.mean(emb_feat_list[i], dim=1))
            
            sim_matrix_seq = sim_matrix[i]
            B = sim_matrix_seq.shape[0]
            N, c, h, w = cur_emb_feat.shape
            cur_emb_feat = cur_emb_feat.reshape(B,-1,c,h,w)
            cur_emb_feat = cur_emb_feat[:,:self.prev_len] if (not use_gt) else cur_emb_feat.clone()

            for t in range(self.aft_seq_length):
                active_matrix_seq = sim_matrix_seq[:,:(self.prev_len-1)]
                if t > 0:
                    fut_t_matrix =sim_matrix_seq[:,(self.prev_len+t-1):(self.prev_len+t)]
                else:
                    fut_t_matrix = sim_matrix_seq[:,(self.prev_len-1):(self.prev_len+t)]
                active_matrix_seq = torch.cat([active_matrix_seq,fut_t_matrix],dim=1)
            
                cur_sim_matrix = cum_multiply(active_matrix_seq.clone())  # B, T+1, h,w,h,w
                composed_t_feats = self.feat_generator(cur_emb_feat[:, :self.prev_len].clone(),
                                                        cur_sim_matrix,feat_idx=i,img_compose=img_compose,scale=scale)
                                                    
                fut_emb_feat_list[i].append(composed_t_feats.clone())
                # update future frame features in the emb_feat_list
                if (not use_gt):
                    if scale == 1:
                        if  cur_emb_feat.shape[1] > self.prev_len+t:
                            cur_emb_feat[:,t+self.prev_len] = composed_t_feats.clone()
                        else:
                            cur_emb_feat = torch.cat([cur_emb_feat,composed_t_feats.clone().unsqueeze(1)],dim=1) #cat compose features for next frame prediction

            temp = torch.stack(fut_emb_feat_list[i], dim=1)
            
            fut_emb_feat_list[i] = temp.reshape(-1, c, h*scale, w*scale) # B*T,c,h,w

        return fut_emb_feat_list,ori_emb_feat_list

    def forward(self, x, similar_matrix, feat_shape):
        compose_feat_list = []
        similar_matrix_for_compose = []
        for i in range(len(x)):
            if x[i] is None:
                compose_feat_list.append(None)
                similar_matrix_for_compose.append(None)
                continue
            if i < len(x) - 2:
                h, w = x[i].shape[-2:]
                target_size = (h // self.feat_scale_list[i] * self.feat_scale_list[i], w // self.feat_scale_list[i] * self.feat_scale_list[i])
                cur_feat = self.feat_unshuffle[i](F.interpolate(x[i].clone(), size=target_size, mode='bilinear'))

                if (cur_feat.shape[-2] != self.mat_size[0][-2]) or (cur_feat.shape[-1] != self.mat_size[0][-1]):
                    compose_feat_list.append(F.interpolate(cur_feat, size=tuple(self.mat_size[0]), mode='bilinear'))
                else:
                    compose_feat_list.append(cur_feat.clone())
                
                similar_matrix_for_compose.append(similar_matrix[0])
            else:
                if (x[i].shape[-2] != self.mat_size[i - len(x) + 2][-2]) or (x[i].shape[-1] != self.mat_size[i - len(x) + 2][-1]):
                    compose_feat_list.append(F.interpolate(x[i], size=tuple(self.mat_size[i - len(x) + 2]), mode='bilinear'))
                else:
                    compose_feat_list.append(x[i])
        
        similar_matrix_for_compose.append(similar_matrix[0])
        similar_matrix_for_compose.append(similar_matrix[1])

        compose_fut_feat_list, _ = self.feat_compose(compose_feat_list, similar_matrix_for_compose)

        for i in range(len(compose_fut_feat_list)):
            if compose_fut_feat_list[i] is None:
                continue
            if i < len(x) - 2:
                compose_fut_feat_list[i] = self.feat_shuffle[i](compose_fut_feat_list[i])
            if (compose_fut_feat_list[i].shape[-2] != feat_shape[i][-2]) or (compose_fut_feat_list[i].shape[-1] != feat_shape[i][-1]):
                compose_fut_feat_list[i] = F.interpolate(compose_fut_feat_list[i], size=tuple(feat_shape[i]), mode='bilinear')

        return compose_fut_feat_list
            


        
class RRDBDecoder(nn.Module):
    def __init__(self, C=1, hid_S=32, downsample_scale=[2,2,2], rrdb_decoder_num=2, scale_in_use=3):
        super(RRDBDecoder, self).__init__()

        self.scale_num = len(downsample_scale)
        out_channel = C * 4 
 
        self.upsample_block_low2high = Up(in_channels=hid_S * (2 ** self.scale_num),
                                          out_channels=hid_S * (2 ** (self.scale_num - 1)),
                                          bilinear=False,
                                          scale=downsample_scale[-1])

        upsample_block_list = []
        for i in range(self.scale_num - 2, -1, -1):
            skip=False if ((i<self.scale_num-1 and scale_in_use == 2) or (i<self.scale_num-2 and scale_in_use == 3)) else True
            upsample_block_list.append(Up(in_channels=hid_S * (2 ** (i+1)),
                                          out_channels=hid_S * (2 ** i),
                                          bilinear=False,
                                          scale=downsample_scale[i],
                                          skip=skip))
        self.upsample_block =  nn.ModuleList(upsample_block_list)

        self.rrdb_block = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_decoder_num)])

        self.outc = nn.Conv2d(hid_S, out_channel, kernel_size=1)

    def forward(self, in_feat):

        x = self.upsample_block_low2high(in_feat[-1], in_feat[-2])
        for i in range(self.scale_num-1):
            x = self.upsample_block[i](x,in_feat[-i-3])

        x = self.rrdb_block(x)
        logits = self.outc(x)
        return logits

class ImageEnhancer(nn.Module):
    def __init__(self, C_in=1, hid_S=32, downsample_scale=[2,2,2], rrdb_enhance_num=2):
        super(ImageEnhancer, self).__init__()
        self.C_in = C_in
        layers = [nn.Conv2d(C_in * 2, hid_S, 3, 1, 1)]
        for i in range(rrdb_enhance_num):
            layers.append(RRDB(hid_S))
        self.model = nn.Sequential(*layers)

        self.outconv = nn.Conv2d(hid_S, C_in, kernel_size=1)

    def forward(self, x,):
        feat = self.model(x)
        out = self.outconv(feat)
        return out
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf):
        super(RRDB, self).__init__()
        gc = nf // 2
        self.RDB1 = ResidualDenseBlock_4C(nf, gc)
        self.RDB2 = ResidualDenseBlock_4C(nf, gc)
        self.RDB3 = ResidualDenseBlock_4C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
class ResidualDenseBlock_4C(nn.Module):
    def __init__(self, nf=64, gc = 32, bias=True):
        super(ResidualDenseBlock_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False,
                 upsample=False, skip=False, factor=2, motion=False):
        super().__init__()
        self.upsample = upsample
        self.maxpool= None
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if factor == 4:
                self.maxpool = nn.MaxPool2d(2)
            
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            
            if motion:
                self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor,
                                                          mode='bilinear',
                                                          align_corners=True),
                                              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                              nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor,
                                                          mode='bilinear',
                                                          align_corners=True),
                                              nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1))

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        input = input + shortcut
        if self.maxpool is not None:
            input = self.maxpool(input)
        return nn.LeakyReLU()(input)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, skip=True, scale=2, bn=True, motion=False):
        super().__init__()
        factor = scale
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if skip:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(in_channels, out_channels, bn=bn)

            else:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(in_channels, out_channels)
        else:
            if skip:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
                self.conv = ConvLayer(out_channels*2, out_channels, bn=bn, motion=motion)
            else:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
                self.conv = ConvLayer(out_channels, out_channels, bn=bn, motion=motion)

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        if x2 is None:
            return self.conv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x) 
    
class PredictModel(nn.Module):
    def __init__(self, T, hidden_len=32, aft_seq_length=10, mx_h=32, mx_w=32, use_direct_predictor=True):
        super(PredictModel, self).__init__()
        self.mx_h = mx_h
        self.mx_w = mx_w
        self.hidden_len = hidden_len
        self.fut_len = aft_seq_length 
        self.conv1 = nn.Conv2d( 1, hidden_len, kernel_size=3, padding=1, bias=False)
        self.fuse_conv = nn.Conv2d(hidden_len*2, hidden_len, kernel_size=3, padding=1, bias=False)
        if use_direct_predictor:
            self.predictor = SimpleMatrixPredictor3DConv_direct(T=T, hidden_len=hidden_len, aft_seq_length=aft_seq_length)
        else:
            self.predictor = MatrixPredictor3DConv(hidden_len)
        self.out_conv = nn.Conv2d(hidden_len, 1, kernel_size=3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def res_interpolate(self,in_tensor,template_tensor):
        '''
        in_tensor: batch,c,h'w',H'W'
        tempolate_tensor: batch,c,hw,HW
        out_tensor: batch,c,hw,HW
        '''
        out_tensor = F.interpolate(in_tensor,template_tensor.shape[-2:]) # (BThw,target_h,target_w)

        return out_tensor

    def forward(self,matrix_seq, softmax=False, res=None):

        B,T,hw,window_size = matrix_seq.size()

        matrix_seq = matrix_seq.reshape(-1,hw,self.mx_h,self.mx_w) # (BT,hw,hw)
        matrix_seq = matrix_seq.reshape(B*T*hw,self.mx_h,self.mx_w).unsqueeze(1) # (BThw,1,h,w)

        x = self.conv1(matrix_seq)
        x = x.reshape(B,T,hw,-1,self.mx_h,self.mx_w)
        x = x.permute(0,2,1,3,4,5).reshape(B*hw,T,-1,self.mx_h,self.mx_w)
        emb = self.predictor(x)

        emb = emb.reshape(B*hw*self.fut_len,-1,self.mx_h,self.mx_w)
        res_emb = emb.clone()
        if res is not None:
            template = emb.clone().reshape(B,hw,emb.shape[1],-1).permute(0,2,1,3)
            in_tensor = res.clone().reshape(B,hw//4,emb.shape[1],-1).permute(0,2,1,3)
            
            res_tensor = self.res_interpolate(in_tensor,template).permute(0,2,1,3).reshape(emb.shape)
            
            emb = self.fuse_conv(torch.cat([emb,res_tensor],dim=1))

        out = self.out_conv(emb) #(Bhwt,16,h//4,w//4)
        
        out = out.reshape(B,hw,-1,self.mx_h,self.mx_w)
        out = out.permute(0,2,1,3,4)
        out = out.reshape(B,-1,hw,window_size)
        
        if softmax:
            out = out.view(B,out.shape[1],-1)
            out = self.softmax(out)
            out = out.reshape(B,-1,hw,window_size)

        return out,res_emb    

class MatrixPredictor3DConv(nn.Module):
    def __init__(self, hidden_len=64):
        super(MatrixPredictor3DConv, self).__init__()
        self.unet_base = hidden_len #64
        self.hidden_len = hidden_len #64
        self.conv_pre_1 = nn.Conv2d(hidden_len,hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)      

        self.conv3d_1 = Conv3D(self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = Conv3D(self.unet_base*2, self.unet_base*2, kernel_size=(3  , 3, 3), stride=1, padding=(0, 1, 1))

        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        
        self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
        self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
        self.bn1_1 = nn.BatchNorm2d(self.unet_base)
        self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
        self.bn3_1 = nn.BatchNorm2d(self.unet_base)
        self.bn4_1 = nn.BatchNorm2d(self.hidden_len)
            
        
    def forward(self,x):
        # x [B,T,C,32,32]
        # out: [B,C,32,32]
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x))) 
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x))) 
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x))) 
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1) #  (batch, seq, c, h, w), 1st temporal conv
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1)))    # (batch * seq, c, h // 2, w // 2)
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2) # (batch, seq=1, c, h // 2, w // 2), 2nd temporal conv
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h//2, w//2), seq = 1
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)) # (batch, seq, c, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous() # (batch, c, seq, h, w)                                           
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None)) # (batch, c, 1, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, c, h, w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous() # (batch*1, c, h, w)
        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1)))) 
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3)) # (batch, seq, 1, h, w)
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, scale_factor=(2, 2)))))         
        return x

class ConvLayer(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=True, motion=False, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )  if motion else  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):
        return self.conv(x)
    
class SimpleMatrixPredictor3DConv_direct(nn.Module):
    def __init__(self, T, hidden_len=64, image_pred=False, aft_seq_length=10):
        super(SimpleMatrixPredictor3DConv_direct, self).__init__()
        self.unet_base = hidden_len #64
        self.hidden_len = hidden_len #64
        self.conv_pre_1 = nn.Conv2d(hidden_len,hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.fut_len = aft_seq_length 

        self.conv3d_1 = Conv3D(self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        
        if self.fut_len > 1 :
            self.temporal_layer = Conv3D(self.unet_base*2, self.unet_base*2, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        else:
            self.temporal_layer = nn.Sequential(
            nn.Conv2d(self.unet_base *2, self.unet_base * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())

        input_len = T if image_pred else T - 1 
        self.conv_translate = nn.Sequential(
            nn.Conv2d(self.unet_base * input_len , self.unet_base * self.fut_len, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        
        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        
        self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
        self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
        self.bn1_1 = nn.BatchNorm2d(self.unet_base)
        self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
        self.bn3_1 = nn.BatchNorm2d(self.unet_base)
        self.bn4_1 = nn.BatchNorm2d(self.hidden_len)
        self.bn_translate = nn.BatchNorm2d(self.unet_base * self.fut_len)
            
        
    def forward(self,x):
        # x [B,T,C,32,32]
        # out: [B,C,32,32]
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x))) 
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x))) 
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x))) 
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        
        x_1 = self.conv3d_1(x_1) #  (batch, seq, c, h, w), 1st temporal conv
        batch, seq, c, h, w = x_1.shape
        x_tmp = x_1.reshape(batch,-1,h,w)
        x_tmp = self.bn_translate(self.conv_translate(x_tmp)) 
        x_1 = x_tmp.reshape(batch,self.fut_len,c,h,w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1))) # (batch * seq, c, h // 2, w // 2)
        if self.fut_len > 1:
            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_2 = self.temporal_layer(x_2) # (batch, seq=10, c, h // 2, w // 2)
        
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h//2, w//2), seq = 1
        else:
            x_2 = self.temporal_layer(x_2) # (batch * seq,c, h // 2, w // 2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)) # (batch, seq, c, h, w)
        
        x_1 = x_1.reshape(-1, x_1.size(2), x_1.size(3), x_1.size(4))


        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, size=x_1.shape[2:]), x_1), dim=1))))
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3)) # (batch, seq, 1, h, w)
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, size = x.shape[3:])))) 
        
        return x

class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.leaky_relu(self.bn3d(self.conv3d(x))) 
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x

def build_similarity_matrix(emb_feats,thre=-1,sigmoid=False,k=-1,cut_off=False):
    '''

    :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)
    :return: similarity matrix (N, T-1, h*w, h*w) current frame --> next frame
    '''
    B,T,c,h,w = emb_feats.shape
    emb_feats = emb_feats.permute(0,1,3,4,2) #  (B,T,h,w,c)
    normalize_feats = emb_feats / (torch.norm(emb_feats,dim=-1,keepdim=True)+1e-6) #  (B,T,h,w,c)
    prev_frame = normalize_feats[:,:T-1].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
    next_frame = normalize_feats[:,1:].reshape(-1,h*w,c,) # (B*(T-1),h*w,c)
    similarity_matrix = torch.bmm(prev_frame,next_frame.permute(0,2,1)).reshape(B,T-1,h*w,h*w) # (N*(T-1)*h*w)
    
    if cut_off:
        similarity_matrix = cut_off_process(similarity_matrix,thre,sigmoid,k)

    return similarity_matrix

def cut_off_process(similarity_matrix,thre,sigmoid=False,k=-1):

    B = similarity_matrix.shape[0]
    T_prime = similarity_matrix.shape[1]
    hw = similarity_matrix.shape[2]
    new_similarity_matrix = similarity_matrix.clone()
    #mask all diagonal to zeros
    '''
    diagonal_mask = torch.zeros_like(new_similarity_matrix[0,0]).to(similarity_matrix.device).bool() #(h*w,h*w)
    diagonal_mask.fill_diagonal_(True)
    diagonal_mask = diagonal_mask.reshape(1,1,hw,hw).repeat(B,T_prime,1,1)
    new_similarity_matrix[diagonal_mask] = 0.
    '''
    if sigmoid:
        new_similarity_matrix[new_similarity_matrix<0] = 0.
        new_similarity_matrix = F.sigmoid(new_similarity_matrix)
        #similarity_matrix = F.sigmoid((similarity_matrix+1)/2.)
    elif k > -1: # select top k
        new_similarity_matrix[new_similarity_matrix<0.] = 0.
        select_num = int(new_similarity_matrix.shape[-1] * k)
        top_k,_ = torch.topk(new_similarity_matrix,select_num,dim=-1)
        thre_value = top_k[:,:,:,-1:]
        new_similarity_matrix[new_similarity_matrix<thre_value] = 0.
    else:
        new_similarity_matrix[new_similarity_matrix<thre] = 0.

    return new_similarity_matrix

def sim_matrix_interpolate(in_matrix,ori_hw,target_hw):

    ori_h,ori_w = ori_hw[0],ori_hw[1]
    target_h,target_w = target_hw[0],target_hw[1]
    B,T,hw,hw = in_matrix.shape
    ori_matrix = in_matrix.clone().reshape(B,T,ori_h,ori_w,ori_h,ori_w)
    ori_matrix_half = F.interpolate(ori_matrix.reshape(-1,ori_h,ori_w).unsqueeze(1),(int(target_h),int(target_w)),mode='bilinear').squeeze(1) # (BThw,target_h,target_w)
    new_matrix = F.interpolate(ori_matrix_half.reshape(B,T,ori_h,ori_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(-1,ori_h,ori_w).unsqueeze(1),(int(target_h),int(target_w)),mode='bicubic').squeeze(1) #(BT*targethw,target_h,target_w)
    new_matrix = new_matrix.reshape(B,T,target_h,target_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(B,T,target_h*target_w,target_h*target_w)

    return new_matrix

def sim_matrix_postprocess(similar_matrix):
    B,T,hw1,hw2 = similar_matrix.shape

    similar_matrix = similar_matrix.reshape(similar_matrix.shape[0],similar_matrix.shape[1],-1)
    similar_matrix = F.softmax(similar_matrix,dim=-1)


    return similar_matrix.reshape(B,T,hw1,hw2)

def cum_multiply(value_seq, cum_softmax = False,reverse=True):
    '''

    :param value_seq: (B,S,***), B - batch num; S- sequence len
    :return: output value_seq: (B,S,***)
    '''
    #print(value_seq.shape)
    if not reverse: # reverse means last element is the one multiplied most times,i.e. the reference is the last element:
        value_seq = torch.flip(value_seq,dims=[1])
    B,T,hw,hw = value_seq.shape
    new_output = value_seq.clone()
    for i in range(value_seq.shape[1]-2,-1,-1):
        cur_sim = new_output[:, i].reshape(B,hw,hw).clone()
        next_sim = new_output[:,i+1].reshape(B,hw,hw).clone()
        new_output[:,i] = torch.bmm(cur_sim,next_sim).reshape(B,hw,hw)
    
    if not reverse:
        new_output = torch.flip(new_output,dims=[1])
    if cum_softmax:
        new_output = sim_matrix_postprocess(new_output)
    return new_output