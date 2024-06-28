import torch
import torch.nn as nn
import torch.nn.functional as F

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:,:self.channel_in,:,:],out[:,self.channel_in:self.channel_in*4,:,:]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


class SRCNN(nn.Module):
    def __init__(self, num_channels, out_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
        self.IDWT = IWT()

    def forward(self, x):
        x = self.DWT(x)
        print(x.shape)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.IDWT(x)
        return x


# model = SRCNN(num_channels=64, out_channels=64).cuda()
# x = torch.randn(1,64,256,256).cuda()
# y = model(x)
# print(y.shape)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# def numpy2tensor(img, rgb_range=1.):
#     img = np.array(img).astype('float64')
#     np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
#     tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
#     tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
#     tensor = tensor.unsqueeze(0)
#     return tensor
# def tensor2numpy(tensor, rgb_range=1.):
#     rgb_coefficient = 255 / rgb_range
#     img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
#     img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)  # CHW -> HWC
#     return img
# dwt = DWT()
# #
# img = cv2.imread("/media/xbm/data/xbm/feat_prop.jpeg")
# img_tensor = numpy2tensor(img)
# img_wave = dwt(img_tensor)
# print(img.shape, img_tensor.shape, img_wave.shape)
# img0 = tensor2numpy(img_wave[0, :, :, :])
# img1 = tensor2numpy(img_wave[1, :, :, :])
# img2 = tensor2numpy(img_wave[2, :, :, :])
# img3 = tensor2numpy(img_wave[3, :, :, :])
# plt.subplot(2, 2, 1)
# plt.imshow(img0[:, :, ::-1])
# plt.subplot(2, 2, 2)
# plt.imshow(img1[:, :, ::-1])
# plt.subplot(2, 2, 3)
# plt.imshow(img2[:, :, ::-1])
# plt.subplot(2, 2, 4)
# plt.imshow(img3[:, :, ::-1])
# plt.show()
# plt.imsave("ll.png", img0[:, :, ::-1])
# plt.imsave("lh.png", img1[:, :, ::-1])
# plt.imsave("hl.png", img2[:, :, ::-1])
# plt.imsave("hh.png", img3[:, :, ::-1])