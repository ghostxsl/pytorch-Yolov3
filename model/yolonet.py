import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

# support route shortcut
class YoLoNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(YoLoNet, self).__init__()

        # Initial parameters
        self.Nchannels = 32
        self.seen = 0
        self.num_classes = num_classes
        self.num_anchors = 3
        self.width = 416
        self.height = 416
        out_channels = (5+self.num_classes) * self.num_anchors

        # Initial convolution layers
        self.conv1_1 = ConvLayer_BN(in_channels, self.Nchannels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ConvLayer_BN(self.Nchannels, self.Nchannels * 2, kernel_size=3, stride=2, padding=1)
        self.block1_1 = ShortcutBlock(self.Nchannels * 2)
        self.conv2 = ConvLayer_BN(self.Nchannels * 2, self.Nchannels * 4, kernel_size=3, stride=2, padding=1)
        self.block2_1 = ShortcutBlock(self.Nchannels * 4)
        self.block2_2 = ShortcutBlock(self.Nchannels * 4)
        self.conv3 = ConvLayer_BN(self.Nchannels * 4, self.Nchannels * 8, kernel_size=3, stride=2, padding=1)
        self.block3_1 = ShortcutBlock(self.Nchannels * 8)
        self.block3_2 = ShortcutBlock(self.Nchannels * 8)
        self.block3_3 = ShortcutBlock(self.Nchannels * 8)
        self.block3_4 = ShortcutBlock(self.Nchannels * 8)
        self.block3_5 = ShortcutBlock(self.Nchannels * 8)
        self.block3_6 = ShortcutBlock(self.Nchannels * 8)
        self.block3_7 = ShortcutBlock(self.Nchannels * 8)
        self.block3_8 = ShortcutBlock(self.Nchannels * 8)
        self.conv4 = ConvLayer_BN(self.Nchannels * 8, self.Nchannels * 16, kernel_size=3, stride=2, padding=1)
        self.block4_1 = ShortcutBlock(self.Nchannels * 16)
        self.block4_2 = ShortcutBlock(self.Nchannels * 16)
        self.block4_3 = ShortcutBlock(self.Nchannels * 16)
        self.block4_4 = ShortcutBlock(self.Nchannels * 16)
        self.block4_5 = ShortcutBlock(self.Nchannels * 16)
        self.block4_6 = ShortcutBlock(self.Nchannels * 16)
        self.block4_7 = ShortcutBlock(self.Nchannels * 16)
        self.block4_8 = ShortcutBlock(self.Nchannels * 16)
        self.conv5 = ConvLayer_BN(self.Nchannels * 16, self.Nchannels * 32, kernel_size=3, stride=2, padding=1)
        self.block5_1 = ShortcutBlock(self.Nchannels * 32)
        self.block5_2 = ShortcutBlock(self.Nchannels * 32)
        self.block5_3 = ShortcutBlock(self.Nchannels * 32)
        self.block5_4 = ShortcutBlock(self.Nchannels * 32)
        self.detect1 = DetectBlock(self.Nchannels * 32, self.Nchannels * 32, out_channels)
        self.upsample1 = UpsampleBlock(self.Nchannels * 16)
        self.detect2 = DetectBlock(self.Nchannels * 16, self.Nchannels * (16+8), out_channels)
        self.upsample2 = UpsampleBlock(self.Nchannels * 8)
        self.detect3 = DetectBlock(self.Nchannels * 8, self.Nchannels * (8 + 4), out_channels)

    def forward(self, x):
        y = self.conv1_1(x)
        y = self.conv1_2(y)
        y = self.block1_1(y)

        y = self.conv2(y)
        y = self.block2_1(y)
        y = self.block2_2(y)

        y = self.conv3(y)
        y = self.block3_1(y)
        y = self.block3_2(y)
        y = self.block3_3(y)
        y = self.block3_4(y)
        y = self.block3_5(y)
        y = self.block3_6(y)
        y = self.block3_7(y)
        route2 = self.block3_8(y)

        y = self.conv4(route2)
        y = self.block4_1(y)
        y = self.block4_2(y)
        y = self.block4_3(y)
        y = self.block4_4(y)
        y = self.block4_5(y)
        y = self.block4_6(y)
        y = self.block4_7(y)
        route1 = self.block4_8(y)

        y = self.conv5(route1)
        y = self.block5_1(y)
        y = self.block5_2(y)
        y = self.block5_3(y)
        y = self.block5_4(y)

        output1, y = self.detect1(y)

        y = self.upsample1(y, route1)
        output2, y = self.detect2(y)

        y = self.upsample2(y, route2)
        output3, y = self.detect3(y)

        return output1, output2, output3

    def load_pretrained_weights(self, weightfile):
        pretrained_params = torch.load(weightfile)
        model_dict = self.state_dict()
        pretrained_params = {k: v for k, v in pretrained_params.items() if k in model_dict}
        self.seen = 0
        model_dict.update(pretrained_params)
        self.load_state_dict(model_dict)
        del pretrained_params
        del model_dict

    def load_weights(self, weightfile):
        params = torch.load(weightfile, map_location=torch.device('cpu'))
        if 'seen' in params.keys():
            self.seen = params['seen']
            del params['seen']
        else:
            self.seen = 0
        self.load_state_dict(params)
        del params

    def save_weights(self, outfile):
        params = self.state_dict()
        params['seen'] = self.seen
        torch.save(params, outfile)
        del params

    def load_binary_weights(self, weightfile):
        pretrained_params = torch.load(weightfile)
        model_dict = self.state_dict()

        key1 = list(pretrained_params.keys())
        key2 = list(model_dict.keys())
        j=0
        for i in range(len(key2)):
            k_1 = key1[j].split('.')
            k_2 = key2[i].split('.')
            if(k_1[-1] == k_2[-1]):
                pretrained_params[key2[i]] = pretrained_params.pop(key1[j])
                j += 1
        pretrained_params = {k: v for k, v in pretrained_params.items() if k in model_dict}
        self.seen = 0
        model_dict.update(pretrained_params)
        self.load_state_dict(model_dict)
        del pretrained_params
        del model_dict

class ConvLayer_BN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer_BN, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv2d(x)))
        return y

class ShortcutBlock(torch.nn.Module):
    def __init__(self, channels, in_channels = None, shortcut = True):
        super(ShortcutBlock, self).__init__()

        self.shortcut =shortcut
        if in_channels is not None:
            self.conv1 = nn.Conv2d(in_channels, int(channels / 2), kernel_size=1, stride=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(channels, int(channels / 2), kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(channels / 2))
        self.conv2 = nn.Conv2d(int(channels / 2), channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, route = False):
        if self.shortcut:
            y = self.leakyrelu(self.bn1(self.conv1(x)))
            y = self.leakyrelu(self.bn2(self.conv2(y)))
            y = y + x
            return y
        elif route:
            y1 = self.leakyrelu(self.bn1(self.conv1(x)))
            y = self.leakyrelu(self.bn2(self.conv2(y1)))
            return y, y1
        else:
            y = self.leakyrelu(self.bn1(self.conv1(x)))
            y = self.leakyrelu(self.bn2(self.conv2(y)))
            return y

class DetectBlock(torch.nn.Module):
    def __init__(self, channels, in_channels, out_channels, shortcut=False):

        super(DetectBlock, self).__init__()
        self.block1 = ShortcutBlock(channels, in_channels, shortcut=False)
        self.block2 = ShortcutBlock(channels, shortcut=shortcut)
        self.block3 = ShortcutBlock(channels, shortcut=False)
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y, y1 = self.block3(y, True)
        y = self.conv(y)
        return y, y1

class UpsampleBlock(torch.nn.Module):
    def __init__(self, channels):

        super(UpsampleBlock, self).__init__()
        self.conv = ConvLayer_BN(channels, int(channels / 2), kernel_size=1, stride=1, padding=0)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, x1):
        y = self.conv(x)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = torch.cat((y, x1), 1)
        return y
