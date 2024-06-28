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
        # extractor & reconstruction
        #self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        #self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)
        ## 
        ## mini_restore
        #restor-￡?n_feats0-kao lv32 or xian 24
        #xian 3bian-24,look xia ,ruhe jinxing
        self.n_feats0=num_feat
        self.n_feats=num_feat
        self.num_feat=num_feat
        self.A1=nn.Sequential(make_layer(ResidualBlockNoBN, 10, num_feat=num_feat))
        self.A2=nn.Sequential(make_layer(ResidualBlockNoBN, 10, num_feat=num_feat))
        self.Block1=nn.Sequential(make_layer(ResidualBlockNoBN, 3, num_feat=num_feat*4))
        self.Block2=nn.Sequential(make_layer(ResidualBlockNoBN, 1, num_feat=num_feat))
        ##μí·?±?ìáì??÷
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        ###to 2D
        self.feat_extract1 = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        #self.orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.conv_last =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.conv_last1 =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.dyd= DynamicDWConv(self.num_feat*16, 3, 1, self.num_feat*16)
        #self.orb2_4 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        #self.orb2_7 = TFR_UNet(self.n_feats0*4, self.n_feats*4, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        #self.orb2_8 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        #self.conv_last2 =  nn.Sequential(nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True),nn.Sigmoid())
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
         #S;lrs_mid_4_00
         #L:lrs_aft 
         pixdown = torch.nn.PixelUnshuffle(4)
         pixup = torch.nn.PixelShuffle(2)
         L_d=pixdown(L) 
         S_d=pixdown(S) 
         #s_d=rearange(b)
         #s_d_weight=self.dyd()
         mid=[]
         ##
         for i in range(b):
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))#[1,384]
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         #mid_feature=F.conv2d(L_d, S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
         mid=torch.stack(mid)
         #print('******mid',mid.shape)
         out=pixup(mid)
         return out
    def model_B2(self,L,S,b):
         pixup = torch.nn.PixelShuffle(4)
         pixdown = torch.nn.PixelUnshuffle(2)
         pixdown2 = torch.nn.PixelUnshuffle(4)
         L_d=pixdown(L)
         #print('******L_d',L_d.shape) 
         S_d=pixdown2(S)
         #print('******S_d',S_d.shape)
         mid=[]
         for i in range(b):
             #print('******S_d1',S_d[i].unsqueeze(0).shape)
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))#[1,384]
             #print('******L_d[i]',L_d[i].unsqueeze(0).shape)   ###
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         #mid_feature=F.conv2d(L_d, S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
         mid=torch.stack(mid)
         #print('******mid',mid.shape)
         out=pixup(mid)
         return out
    def forward(self, lrs):
        b, t, c, h, w = lrs.size()
        # time_start = time.time()
        # print(lrs.size())
        # 1lrsè??D??2?ì??÷
        lrs_mid=lrs[:,t//2]#btchw￡?è??D??t//2
        #print("lrs_mid is what",lrs_mid[:,0,0,0]*2.56)
        #print("fenshu*********",fenshu)
        #jiangcaiyang *4 =128*128
        #scale_factor = (1, 1, 0.25, 0.25) //
        ##mid down 4 
        #
        #lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=scale_factor, mode='bilinear',align_corners=False)
        lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=0.25, mode='bilinear', align_corners=False)
        #ìáè? lrs_mid_4ì??÷  
        lrs_mid_4_0 = self.feat_extract(lrs_mid_4) # 3±?24
        lrs3d=self.feat_extract1(rearrange(lrs, 'b t c h w -> (b t) c h w')) 
        #pass by stage00 
        lrs3d=rearrange(lrs3d, '(b t) c h w -> b t c h w',b=b)
        lrs_mid_4_00=self.stageA00(lrs_mid_4_0)  #
        #lrs3d = rearrange(lrs3d,'b c t h w -> b t c h w')
        #keep lrs3d =bt
        #Large 3image fea extrac   
        #change 3d to 2d 
        #lrs3d=self.feat_extractor(rearrange(lrs, 'b t c h w -> (b t) c h w'))
        #print('lrs3d.shape',lrs3d.shape)
        ###3 dim feature map div
        #piexunshuffle???ü????￡???á?
        #qingyibiegai fout=self.model_B(lrs3d,lrs_mid_4_00,b,t)
        #lrs3d=bt,will change
        for i in range(t):
            lrs_=lrs3d[:,i]
            f_ele=self.model_B(lrs_,lrs_mid_4_00,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        
                #[2, 3, 96, 512, 512]
        #print('#############f_out',f_out.shape)
        ##
        lrs3d_B = rearrange(f_out, 'b t c h w -> (b t) c h w')
        lrs3d_StageB1 = self.stageB10(lrs3d_B)  ##pass by [3,3]conv
        #lrs3d_StageB1=self.stageB10(lrs3d_B) #pass by [1,3,3] conv
        #print('#############lrs3d_StageB1',lrs3d_StageB1)
        #lrs_mid?-1y4??TFR_UNetμ?μ??????á1?,ì??÷ò?2?·?è￥èúo?￡?ò?2?·?è￥éú3éGT
        ##### 
        delivery_4_fea,restore_4_out= self.stageA0(lrs_mid_4_00)
        lrs3d_B2 = rearrange(lrs3d_StageB1, '(b t) c h w -> b t c h w',b=b)
        #####
        for i in range(t):
            lrs_=lrs3d_B2[:,i]
            f_ele=self.model_B2(lrs_,delivery_4_fea,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        #print('#############f_out2',f_out.shape)
        lrs3d_B2 = rearrange(f_out, 'b t c h w -> (b t) c h w')
        ##×?±??-1ystageB11()   
        lrs3d_StageB1=self.stageB11(lrs3d_B2)#b*t c h w'
        output=rearrange(lrs3d_StageB1, '(b t) c h w -> b t c h w',b=b)
        # f1=self.model_B2(f1_out,delivery_4_fea)
        # f2=self.model_B2(f2_out,delivery_4_fea)
        # f3=self.model_B2(f3_out,delivery_4_fea)
        # f1_out=self.stageB1(f1).unsqueeze(1)
        # f2_out=self.stageB1(f2).unsqueeze(1)
        # f3_out=self.stageB1(f3).unsqueeze(1) 
        ##delivery_4_fea??μY?a?a?é??ì??÷??μ??í?yéú3é￡?restore_4_outá?×?è￥????loss
        ##coreweightó?3?????èμ?ì??÷í???DD?í?y￡?????D??￠ìáè?á?￡?????32???í?y2?1?￡??aD??￠ì?éùá??￡
        # output=torch.cat((f1_out,f2_out,f3_out),dim=1)
        #print('restore_4_out.shape()',restore_4_out.shape)
        #print('output.shape()',output.shape)
        return output,restore_4_out##########outputèy???óí?μ??????á1?￡?restore_4_outD?í??????á1?
        #x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups) 
        #lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))     # b c t h w
        #//èyí?μàì??÷ìáè?oó￡?òa2?òa?èpixel shuff￡?í??1í?μàêy￡? ?±?ó?ù?Y?é??ì??÷ìáê???DD?í?y￡?oóD??ù??DDrefine
        #sam_features0, sam_features = self.stage0(x0)
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

