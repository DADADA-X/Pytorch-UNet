# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 普通bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 因为res要和out相加，如果x维度和output不同（分别对应inplane和plane*4），就要对这个x进行降维操作
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 空洞卷积
class BottleneckA(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        # assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 空洞卷积+跳层1x1卷积
class BottleneckB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        # assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# decoder block
class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DecoderBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes//4),
            nn.ReLU(inplace=True),
            # 反卷积的外面一定会补k-1的0，如果padding>0，那补0的个数变小，变为k-1-p；stride控制是空洞大小，s-1；output
            nn.ConvTranspose2d(inplanes//4, inplanes//4, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False),
            nn.BatchNorm2d(inplanes//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes//4, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.up(x)
        return out

# decoder block add
class DecoderBlockAdd(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DecoderBlockAdd, self).__init__()
        self.up = DecoderBlock(inplanes, planes, stride)

    def forward(self, x1, x2):
        out = self.up(x1)
        out = x2 + out
        return out







# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         '''self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3),
#             nn.ReLU(inplace=True)
#         )'''
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),  # 原文有说要加bn吗？没有
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class inconv(nn.Module):  # input的conv块（由double_conv：conv->BN->relu->conv->BN->relu组成）
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )
#
#     def forward(self, x):
#         x = self.mpconv(x)
#         return x
#
#
# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()

#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
#             '''self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)'''

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]  # height
#         diffX = x2.size()[3] - x1.size()[3]  # width

#         # x2 = x2[:, :, diffY // 2: x2.size()[2] - (diffX - diffX // 2),
#         #      diffY // 2: x1.size()[3] - (diffY - diffY // 2)]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))

#         # for padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x
#
#
# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
