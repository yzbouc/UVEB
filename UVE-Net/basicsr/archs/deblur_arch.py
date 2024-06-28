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
class Deblur(nn.Module):
    def __init__(self, num_feat=32, num_block=15):
        super().__init__()
        # extractor & reconstruction
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        #self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)
        ## 
        ## mini_restore
        #restor-，n_feats0-kao lv32 or xian 24
        #xian 3bian-24,look xia ,ruhe jinxing
        self.n_feats0=num_feat
        self.n_feats=num_feat
        self.num_feat=num_feat
        self.A1=nn.Sequential(make_layer(ResidualBlockNoBN, 30, num_feat=num_feat))
        self.A2=nn.Sequential(make_layer(ResidualBlockNoBN, 30, num_feat=num_feat))
        self.Block1=nn.Sequential(make_layer(ResidualBlockNoBN2D, 15, num_feat=num_feat*4))
        self.Block2=nn.Sequential(make_layer(ResidualBlockNoBN2D, 5, num_feat=num_feat))
        ##低分辨提特征
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        #self.orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.conv_last =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.dyd= DynamicDWConv(self.num_feat*16, 3, 1, self.num_feat*16)
        #self.orb2_4 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        #self.orb2_7 = TFR_UNet(self.n_feats0*4, self.n_feats*4, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        #self.orb2_8 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.conv_last2 =  nn.Sequential(nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True),nn.Sigmoid())
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
        x_out = self.conv_last2(x0)
        return x_out
    def model_B(self,L,S,b):
         #S;lrs_mid_4_00
         #L:lrs_aft 
         pixdown = torch.nn.PixelUnshuffle(4)
         pixup = torch.nn.PixelShuffle(2)
         L_d=pixdown(L) 
         S_d=pixdown(S) 
         mid=[]
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
        # 1lrs取中间层特征
        lrs_mid=lrs[:,t//2]#btchw，取中间t//2
        #print("lrs_mid is what",lrs_mid[:,0,0,0]*2.56)
        #print("fenshu*********",fenshu)
        #jiangcaiyang *4 =128*128
        #scale_factor = (1, 1, 0.25, 0.25) //
        ##mid down 4  
        #lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=scale_factor, mode='bilinear',align_corners=False)
        lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=0.25, mode='bilinear', align_corners=False)
        #提取 lrs_mid_4特征  
        lrs_mid_4_0 = self.feat_extract(lrs_mid_4) # 3变24
        #pass by stage00 
        lrs_mid_4_00=self.stageA00(lrs_mid_4_0)  #主要这时候也是128了，这个效果感觉就打折扣
        #Large 3image fea extrac   
        lrs3d=self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))
        #print('lrs3d.shape',lrs3d.shape)
        lrs3d = rearrange(lrs3d,'b c t h w -> b t c h w')
        ###3 dim feature map div
        #piexunshuffle只能四维，算了
        for i in range(t):
            lrs_=lrs3d[:,i]
            f_ele=self.model_B(lrs_,lrs_mid_4_00,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
                #[2, 3, 96, 512, 512]
        #print('#############f_out',f_out.shape)
        ##用三维卷积进行处理，应该速度更快
        lrs3d_B = rearrange(f_out, 'b t c h w -> b c t h w')
        lrs3d_StageB1=self.stageB10(lrs3d_B) #pass by [1,3,3] conv
        #print('#############lrs3d_StageB1',lrs3d_StageB1)
        lrs3d_B2 = rearrange(lrs3d_StageB1, 'b c t h w -> b t c h w')
        # lrs_bef=lrs3d[:3,0]
        # #print('lrs_bef.shape',lrs_bef.shape)
        # lrs_aft=lrs3d[:3,2]
        # lrs_mid_3=lrs3d[:3,1]
        ##lrs_mid00 and,b,m,aftpass to B model 
        # f1=self.model_B(lrs_bef,lrs_mid_4_00)
        # f2=self.model_B(lrs_mid_3,lrs_mid_4_00)
        # f3=self.model_B(lrs_aft,lrs_mid_4_00)
        #f1,f2,f3 passby stage10
        # f1_out=self.stageB10(f1)
        # f2_out=self.stageB10(f2)
        # f3_out=self.stageB10(f3)
        #lrs_mid经过4个TFR_UNet得到恢复结果,特征一部分去融合，一部分去生成GT
        #####
        delivery_4_fea,restore_4_out= self.stageA0(lrs_mid_4_00)
        #####准备获取第二次的卷积核并卷积
        for i in range(t):
            lrs_=lrs3d_B2[:,i]
            f_ele=self.model_B2(lrs_,delivery_4_fea,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        #print('#############f_out2',f_out.shape)
        lrs3d_B2 = rearrange(f_out, 'b t c h w -> b c t h w')
        ##准备经过stageB11() 
        lrs3d_StageB1=self.stageB11(lrs3d_B2)#b c t h w'
        output=rearrange(lrs3d_StageB1, 'b c t h w -> b t c h w')
        # f1=self.model_B2(f1_out,delivery_4_fea)
        # f2=self.model_B2(f2_out,delivery_4_fea)
        # f3=self.model_B2(f3_out,delivery_4_fea)
        # f1_out=self.stageB1(f1).unsqueeze(1)
        # f2_out=self.stageB1(f2).unsqueeze(1)
        # f3_out=self.stageB1(f3).unsqueeze(1) 
        ##delivery_4_fea传递——干净特征指导卷积生成，restore_4_out留着去计算loss
        ##coreweight与3个维度的特征图进行卷积，考虑信息提取量，仅仅32个卷积不够，这信息太少了。
        # output=torch.cat((f1_out,f2_out,f3_out),dim=1)
        #print('restore_4_out.shape()',restore_4_out.shape)
        #print('output.shape()',output.shape)
        return output,restore_4_out##########output三张大图的增强结果，restore_4_out小图增强结果
        #x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups) 
        #lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))     # b c t h w
        #//三通道特征提取后，要不要先pixel shuff，拓展通道数？ 直接根据干净特征提示进行卷积，后续再进行refine
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

