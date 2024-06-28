import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer,ResidualBlockNoBN2D
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from .newDyD import DynamicDWConv as DynamicDWConv
from einops import rearrange
@ARCH_REGISTRY.register()
class Deblur_samll(nn.Module):
    def __init__(self, num_feat=32, num_block=15):
        super().__init__()
        self.n_feats0=num_feat
        self.n_feats=num_feat
        self.num_feat=num_feat
        self.A1=nn.Sequential(make_layer(ResidualBlockNoBN, 10, num_feat=num_feat))
        self.A2=nn.Sequential(make_layer(ResidualBlockNoBN, 10, num_feat=num_feat))
        self.Block1=nn.Sequential(make_layer(ResidualBlockNoBN, 3, num_feat=num_feat*4))
        self.Block2=nn.Sequential(make_layer(ResidualBlockNoBN, 1, num_feat=num_feat))
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        self.feat_extract1 = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        self.conv_last =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.conv_last1 =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.dyd= DynamicDWConv(self.num_feat*16, 3, 1, self.num_feat*16)
        self.bias = nn.Parameter(torch.zeros(self.num_feat*16))
    def stageA00(self,x0):
        x0=self.A1(x0)
        return x0
    def stageA0(self,x0):
        x0=self.A2(x0)
        x_out = self.conv_last(x0)
        return x0,x_out
    def stageB10(self,x0):
        x0=self.Block1(x0)
        return x0
    def stageB11(self,x0):
        x0=self.Block2(x0)
        x_out = self.conv_last1(x0)
        return x_out
    def model_B(self,L,S,b):
         pixdown = torch.nn.PixelUnshuffle(4)
         pixup = torch.nn.PixelShuffle(2)
         L_d=pixdown(L) 
         S_d=pixdown(S) 
         mid=[]
         for i in range(b):
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         mid=torch.stack(mid)
         out=pixup(mid)
         return out
    def model_B2(self,L,S,b):
         pixup = torch.nn.PixelShuffle(4)
         pixdown = torch.nn.PixelUnshuffle(2)
         pixdown2 = torch.nn.PixelUnshuffle(4)
         L_d=pixdown(L)
         S_d=pixdown2(S)
         mid=[]
         for i in range(b):
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         mid=torch.stack(mid)
         out=pixup(mid)
         return out
    def forward(self, lrs):
        b, t, c, h, w = lrs.size()
        lrs_mid=lrs[:,t//2]
        lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=0.25, mode='bilinear', align_corners=False)
        lrs_mid_4_0 = self.feat_extract(lrs_mid_4) 
        lrs3d=self.feat_extract1(rearrange(lrs, 'b t c h w -> (b t) c h w')) 
        lrs3d=rearrange(lrs3d, '(b t) c h w -> b t c h w',b=b)
        lrs_mid_4_00=self.stageA00(lrs_mid_4_0) 
        for i in range(t):
            lrs_=lrs3d[:,i]
            f_ele=self.model_B(lrs_,lrs_mid_4_00,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        lrs3d_B = rearrange(f_out, 'b t c h w -> (b t) c h w')
        lrs3d_StageB1 = self.stageB10(lrs3d_B) 
        delivery_4_fea,restore_4_out= self.stageA0(lrs_mid_4_00)
        lrs3d_B2 = rearrange(lrs3d_StageB1, '(b t) c h w -> b t c h w',b=b)
        for i in range(t):
            lrs_=lrs3d_B2[:,i]
            f_ele=self.model_B2(lrs_,delivery_4_fea,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        lrs3d_B2 = rearrange(f_out, 'b t c h w -> (b t) c h w') 
        lrs3d_StageB1=self.stageB11(lrs3d_B2)#b*t c h w'
        output=rearrange(lrs3d_StageB1, '(b t) c h w -> b t c h w',b=b)
        return output,restore_4_out
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=bias),
                #nn.ReLU(inplace=True),
                #nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
class TFR_UNet(nn.Module):
    def __init__(self, n_feat0, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(TFR_UNet, self).__init__()
        scale_unetfeats = 12
        self.encoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level3 = [CAB(n_feat0 + 2*scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat0, scale_unetfeats)
        self.down23 = DownSample(n_feat0+scale_unetfeats, scale_unetfeats)

        self.decoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level3 = [CAB(n_feat0+scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat0, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat0 + scale_unetfeats, scale_unetfeats)
    def forward(self, x):
        shortcut = x
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
        #                           nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
        # self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True),
        #                    nn.PReLU())
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        x = self.down(x)
        return x
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

