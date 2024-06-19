import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



def make_model(args, parent=False):
    return RCAN(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class ARG:
    def __init__(self, n_resgroups = 10, n_resblocks = 20, n_feats = 64,
                 res_scale = 1, reduction = 16, scale = 4, n_colors = 3,
                 rgb_range = 255, weights_path = '', input_dir = '',
                 outputs_dir = ''):

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.scale = [scale]
        self.res_scale = res_scale
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.weights_path = weights_path
        self.outputs_dir = outputs_dir
        self.input_dir = input_dir




def load_model(opt):
    model = RCAN(opt)

    model.load_state_dict(torch.load(opt.weights_path, map_location='cpu'))
    # model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model

def process_image(model, image_path, output_path, scale):
    input = pil_image.open(image_path).convert('RGB')
    lr = input.resize((input.width // scale, input.height // scale), pil_image.BICUBIC)

    # bicubic = lr.resize((input.width, input.height), pil_image.BICUBIC)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # bicubic.save(output_path.replace('.tiff', '_bicubic.tiff'))

    input = transforms.ToTensor()(lr).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input)

    output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    output = pil_image.fromarray(output, mode='RGB')
    output.save(output_path)

def find_images(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main(opt):
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    model = load_model(opt)

    opt.scale = opt.scale[0]

    # image_paths = find_images(opt.input_dir)[:10]
    image_paths = ['/Users/sachethkoushik/Desktop/RCAN processing/pred_train.tiff']
    for image_path in image_paths:
        relative_path = os.path.relpath(image_path, opt.input_dir)
        output_path = os.path.join(opt.outputs_dir, relative_path)
#         output_path = output_path.replace('.jpeg', '.png').replace('.jpg', '.png').replace('.bmp', '.png')
#         output_path = output_path.replace

        process_image(model, image_path, output_path, opt.scale)
        print(f"Processed {image_path} -> {output_path}")

if __name__ == "__main__":
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = ARG(scale = 2, weights_path = '/Users/sachethkoushik/Desktop/RCAN processing/RCAN_model.pt',
           input_dir = '/Users/sachethkoushik/Desktop/RCAN processing/ICIP training data/0/RawDataQA (1)',
           outputs_dir = 'output/')

    main(opt)