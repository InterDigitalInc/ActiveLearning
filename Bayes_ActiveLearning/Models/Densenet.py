'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralLayersJeffrey import LinearGroupNJ, Conv2dGroupNJ


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, is_cuda=False):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2dGroupNJ(in_planes, 4*growth_rate, kernel_size=1, bias=False, cuda=is_cuda)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = Conv2dGroupNJ(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False, cuda=is_cuda)
        self.bottleneck_kl = [self.conv1, self.conv2]

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, is_cuda=False):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = Conv2dGroupNJ(in_planes, out_planes, kernel_size=1, bias=False, cuda=is_cuda)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, is_cuda=False):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = Conv2dGroupNJ(3, num_planes, kernel_size=3, padding=1, bias=False, cuda=is_cuda)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], is_cuda)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, is_cuda)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], is_cuda)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, is_cuda)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], is_cuda)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, is_cuda)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], is_cuda)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = LinearGroupNJ(num_planes, num_classes, cuda=is_cuda)
        self.kl_list = []
        for m in self.modules():
            if isinstance(m, Conv2dGroupNJ):
                self.kl_list.append(m)
            elif isinstance(m, LinearGroupNJ):
                self.kl_list.append(m)
            elif isinstance(m, Bottleneck):
                self.kl_list += m.bottleneck_kl

    def _make_dense_layers(self, block, in_planes, nblock, is_cuda=False):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, is_cuda))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

def DenseNet100(num_classes, is_cuda):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=num_classes, is_cuda=is_cuda)

def DenseNet121(num_classes, is_cuda):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes, is_cuda=is_cuda)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201(num_classes, is_cuda):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_classes=num_classes, is_cuda=is_cuda)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)


