# full assembly of the sub-parts to form the complete net

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *

class DetNet(nn.Module):
    def __init__(self, num_classes=1):
        self.inplanes = 64
        layers = [3, 4, 6, 3, 3]
        block = Bottleneck

        super(DetNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_new_layer(256, layers[3])
        self.conv6 = self._make_new_layer(256, layers[4])

        self.deconv6 = DecoderBlockAdd(256, 256)
        self.deconv5 = DecoderBlockAdd(256, 1024)
        self.deconv4 = DecoderBlockAdd(1024, 512, stride=2)
        self.deconv3 = DecoderBlockAdd(512, 256, stride=2)
        self.deconv2 = DecoderBlock(256, 64)
        # self.deconv1 = DecoderBlock(64, 64, stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(256, num_classes)

        # 给conv层和bn层初始权重？
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_new_layer(self, planes, blocks):
        downsample = None
        block_b = BottleneckB
        block_a = BottleneckA

        layers = []
        layers.append(block_b(self.inplanes, planes, stride=1, downsample=downsample))
        self.inplanes = planes * block_b.expansion
        for i in range(1, blocks):
            layers.append(block_a(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # down
        c1 = self.conv1(x)
        c2 = self.maxpool(c1)
        c2 = self.conv2(c2)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        # up
        out = self.deconv6(c6, c5)
        out = self.deconv5(out, c4)
        out = self.deconv4(out, c3)
        out = self.deconv3(out, c2)
        out = self.deconv2(out)
        out = self.deconv1(out)
        if out.size()[1] == 1:
            return torch.sigmoid(out)
        else:
            return F.softmax(out, dim=1)


        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#
#         '''self.down4 = down(512, 1024)
#         self.up1 = up(1024, 512, None)
#         self.up2 = up(512, 256, None)
#         self.up3 = up(256, 128, None)
#         self.up4 = up(128, 64, None)'''
#         self.down4 = down(512, 512)         # 这里的output_channel有一点点confusing - 乱写的
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         if x.size()[1] == 1:
#             return torch.sigmoid(x)
#         else:
#             return F.softmax(x, dim=1)
