'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        init.kaiming_normal(self.conv1.weight)
        init.kaiming_normal(self.conv2.weight)
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace = True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out, inplace = True)
        return out

class CNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CNN, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        init.kaiming_normal(self.conv1.weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace = True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def CNN20():
    return CNN(BasicBlock, [3,3,3])

def CNN56():
    return CNN(BasicBlock, [9,9,9])

def CNN110():
    return CNN(BasicBlock, [18, 18, 18])    

def test():
    net = CNN20()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
