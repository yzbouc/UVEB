import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
import .newDyD.DynamicDWConv as DynamicDWConv
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
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
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
class Deblur(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        # extractor & reconstruction
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)
        ##mini_restore
        #down1
        #restor-，n_feats0-kao lv32 or xian 24
        #xian 3bian-24,look xia ,ruhe jinxing
        self.n_feats0=32
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1),CAB(self.n_feats0, 3, 4, bias=False, act=nn.PReLU()))
        self.orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.orb2 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.orb3 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        # self.orb4 =
        self.orb4 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.conv_last =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.dyd= DynamicDWConv(hidden_features, 3, 1, hidden_features)
        ##
        self.1orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.1orb2 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
    def stage0(self, x0):
        shortcut = x0
        x0 = self.orb1(x0)
        x0 = self.orb2(x0)
        x0 = self.orb3(x0)
        x0 = self.orb4(x0)
        x_out = self.conv_last(x0)
        return x0,x_out
        ##
        # res0 = self.orb5(x0)
        # res0 = res0 + shortcut
    def stage1(self, x0):
        shortcut = x0
        x0 = self.1orb1(x0)
        x0 = self.1orb2(x0)
        x_out = self.conv_last(x0)
        return x_out
    def forward(self, lrs):
        b, t, c, h, w = lrs.size()
        # time_start = time.time()
        # print(lrs.size())
        # 1lrs取中间层特征
        lrs_mid=lrs[:3,1]#btchw
        #jiangcaiyang *4 =256,256
        lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=0.25, mode='bilinear', align_corners=False)
        #提取 lrs_mid_4特征
        lrs_mid_4_0 = self.feat_extract(lrs_mid_4) # 3变24
        #lrs_mid经过4个TFR_UNet得到恢复结果,特征一部分去融合，一部分去生成GT
        delivery_4_fea,restore_4_out= self.stage0(lrs_mid_4_0)
        ##delivery_4_fea传递——干净特征指导卷积生成，restore_4_out留着去计算loss
        delivery_4_fea_coreweight=self.dyd(delivery_4_fea)
        ##coreweight与3个维度的特征图进行卷积，考虑信息提取量，仅仅32个卷积不够，这信息太少了。
        ###3 dim feature map div
        lrs_bef=lrs[:3,0]
        lrs_aft=lrs[:3,2]
        f1=F.conv2d(lrs_bef, delivery_4_fea_coreweight, self.bias.repeat(b), stride=1, padding=1, groups=32)
        f2=F.conv2d(lrs_mid, delivery_4_fea_coreweight, self.bias.repeat(b), stride=1, padding=1, groups=32)
        f3=F.conv2d(lrs_aft, delivery_4_fea_coreweight, self.bias.repeat(b), stride=1, padding=1, groups=32)
        #f1,f2,f3 passby 2FUet
        f1_out=self.stage0(f1)
        f2_out=self.stage0(f2)
        f3_out=self.stage0(f3)
        ###kaolv f1_out conca 
        output=torch.cat((f1_out,f2_out,f3_out),dim=1)
        return output,restore_4_out
        #x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        #lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))     # b c t h w
       # //三通道特征提取后，要不要先pixel shuff，拓展通道数？ 直接根据干净特征提示进行卷积，后续再进行refine

        #sam_features0, sam_features = self.stage0(x0)


