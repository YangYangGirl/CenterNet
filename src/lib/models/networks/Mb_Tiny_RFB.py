import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
from torchvision.models.utils import load_state_dict_from_url
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
                      dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class Mb_Tiny_RFB(nn.Module):

    def __init__(self, heads, head_conv, num_classes=2, input_size=224, width_mult=1.):
        super(Mb_Tiny_RFB, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            #conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            #conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        )

        # add heads

        self.inplanes = 128

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(self.inplanes, head_conv,
                              kernel_size=5, padding=2, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(self.inplanes, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        self.fc = nn.Linear(1024, num_classes)
    

    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.cem3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            address = "/home/yy/github/CenterNet/exp/pretrained/shufflenetv2_x0.5_60.646_81.696.pth.tar"
            pretrained_state_dict = torch.load(address)
            self.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.model(x)

        ret = {}

        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

        return x


def get_Mb_Tiny_RFB_net(num_layers, heads, head_conv):
    model = Mb_Tiny_RFB(heads, head_conv, width_mult=0.5)
    model.init_weights(pretrained=False)

    return model
