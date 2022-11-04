import os

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models

import numpy as np

from quan.utils import find_modules_to_quantize, replace_module_by_names
from utils import util, config

from compressai.zoo.image import model_architectures as architectures

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--lsq", action="store_false", help="Apply LSQ quantization"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=1,
        help="Quality (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="quantization config file dir"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    args = parse_args(argv)
    device = "cpu"

    net = models[args.model](quality=args.quality)

    if args.lsq:
        quant_config = config.get_config(args.config)
        modules_to_replace = find_modules_to_quantize(net, quant_config.quan)
        net = replace_module_by_names(net, modules_to_replace)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        if not args.lsq:
            net = architectures[args.model].from_state_dict(checkpoint['state_dict'])
            net.update()
        else:
            net.load_state_dict(checkpoint['state_dict'])

    net = net.to(device)
    net.eval()

    # weight
    '''
    lr = 0
    ch = 20
    weight = net.g_s[lr].weight
    quant = net.g_s[lr].quan_w_fn(weight).data[ch].view(-1).numpy()
    weight = weight.data[ch].view(-1).numpy()
    print(np.abs(weight).max() * 2)
    print(net.g_s[lr].quan_w_fn.s[ch].data * 255)

    plt.subplot(2, 1, 1)
    plt.hist(weight, bins=100)

    plt.subplot(2, 1, 2)
    plt.hist(quant, bins=100)

    plt.show()
    '''

    # activation 분포

    transform = transforms.Compose(
        [transforms.RandomCrop((256, 256)), transforms.ToTensor()]
    )

    tensor = None
    for img_name in os.listdir('c:/kodak'):
        if img_name.endswith(".png"):
            img = Image.open('C:/Kodak/' + img_name).convert('RGB')
            img_t = transform(img).unsqueeze(0)
            if tensor is None:
                tensor = img_t
            else:
                tensor = torch.cat((tensor, img_t), dim=0)

    act = net.g_a[0:4](tensor)
    act = act.reshape(-1)
    act = act.detach().numpy()
    plt.hist(act, bins=100)
    plt.semilogy()
    plt.show()

    # distribution before/after quantization
    '''
    # net.g_a[3](net.g_a[2](net.g_a[5](net.g_a[4]( ))))
    act = net.g_a[1](net.g_a[0](tensor1))
    quant = net.g_a[2].quan_a_fn(act).data.view(-1).numpy()
    act = act.data.view(-1).numpy()

    print(np.abs(act).max())
    print(net.g_a[2].quan_a_fn.s)
    print(np.mean(act))

    plt.subplot(2, 1, 1)
    plt.hist(act, bins=300)

    plt.subplot(2, 1, 2)
    plt.hist(quant, bins=200)

    plt.show()
    '''

    # 각 layer에서 quantization error, step size 출력

    '''
    y0 = tensor
    y1 = net.g_a[0:2](tensor)
    y2 = net.g_a[0:4](tensor)
    y3 = net.g_a[0:6](tensor)
    y = net.g_a(tensor)

    z1 = net.h_a[0:2](y)
    z2 = net.h_a[0:4](y)
    z = net.h_a(y)

    z_hat, z_likelihoods = net.entropy_bottleneck(z)
    gaussian_params = net.h_s(z_hat)
    gp1 = net.h_s[0:2](z_hat)
    gp2 = net.h_s[0:4](z_hat)

    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    y_hat, y_likelihoods = net.gaussian_conditional(y, scales_hat, means=means_hat)

    x1 = net.g_s[0:2](y_hat)
    x2 = net.g_s[0:4](y_hat)
    x3 = net.g_s[0:6](y_hat)
    x_hat = net.g_s(y_hat)

    all_step = 0
    steps = []
    steps_gs = []
    norm = 0
    norms = []
    norms_gs = []
    print('\ng_a')
    i = 0
    for yn, fn in zip([y0, y1, y2, y3],
                      [net.g_a[0].quan_a_fn, net.g_a[2].quan_a_fn, net.g_a[4].quan_a_fn, net.g_a[6].quan_a_fn]):
        quan = fn(yn).detach().view(-1).numpy()
        yn = yn.detach().view(-1).numpy()
        print(i)
        i += 1
        print(np.linalg.norm(yn - quan, 1), ' | ', fn.s.data.item())
        all_step += fn.s.data.item()
        steps.append(fn.s.data.item())
        norm += np.linalg.norm(yn - quan, 1)
        norms.append(np.linalg.norm(yn - quan, 1))

    print('\nh_a')
    i = 0
    for zn, fn in zip([y, z1, z2], [net.h_a[0].quan_a_fn, net.h_a[2].quan_a_fn, net.h_a[4].quan_a_fn]):
        quan = fn(zn).detach().view(-1).numpy()
        zn = zn.detach().view(-1).numpy()
        print(i)
        i += 1
        print(np.linalg.norm(zn - quan, 1), ' | ', fn.s.data.item())
        all_step += fn.s.data.item()
        steps.append(fn.s.data.item())
        norm += np.linalg.norm(zn - quan, 1)
        norms.append(np.linalg.norm(zn - quan, 1))

    print('\nh_s')
    i = 0
    for gp, fn in zip([z_hat, gp1, gp2], [net.h_s[0].quan_a_fn, net.h_s[2].quan_a_fn, net.h_s[4].quan_a_fn]):
        quan = fn(gp).detach().view(-1).numpy()
        gp = gp.detach().view(-1).numpy()
        print(i)
        i += 1
        print(np.linalg.norm(gp - quan, 1), ' | ', fn.s.data.item())
        all_step += fn.s.data.item()
        steps.append(fn.s.data.item())
        norm += np.linalg.norm(gp - quan, 1)
        norms.append(np.linalg.norm(gp - quan, 1))

    print('\ng_s')
    i = 0
    step = 0
    for xn, fn in zip([y_hat, x1, x2, x3],
                      [net.g_s[0].quan_a_fn, net.g_s[2].quan_a_fn, net.g_s[4].quan_a_fn, net.g_s[6].quan_a_fn]):
        quan = fn(xn).detach().view(-1).numpy()
        xn = xn.detach().view(-1).numpy()
        print(i)
        i += 1
        print(np.linalg.norm(xn - quan, 1), ' | ', fn.s.data.item())
        all_step += fn.s.data.item()
        step += fn.s.data.item()
        steps.append(fn.s.data.item())
        steps_gs.append(fn.s.data.item())
        norm += np.linalg.norm(xn - quan, 1)
        norms.append(np.linalg.norm(xn - quan, 1))
        norms_gs.append(np.linalg.norm(xn - quan, 1))

    print('g_s.0 step size', steps[10])
    print('h_s.0 step size', steps[7])
    print('g_s.0 error', norms[10])
    print('h_s.0 error', norms[7])

    print('norm과 step size 상관계수', np.corrcoef(np.array(steps), np.array(norms)))
    print('g_s에서 상관계수', np.corrcoef(np.array(steps_gs), np.array(norms_gs)))

    print('Quantization error L1 norm 합', norm)

    print('전체 step size 평균', all_step / 14)

    print('g_s step size 평균', step / 4)

    print('z_hat 실제 범위, 예측 범위 크기')
    print(z_hat.max() - z_hat.min())
    print(net.h_s[0].quan_a_fn.s.data * 255)

    print('y_hat 실제 범위, 예측 범위 크기')
    print(y_hat.max() - y_hat.min())
    print(net.g_s[0].quan_a_fn.s.data * 255)

    plt.subplot(2, 1, 1)
    plt.hist(tensor.detach().view(-1).numpy(), bins=255)
    plt.subplot(2, 1, 2)
    plt.hist(net.g_a[0].quan_a_fn(tensor).detach().view(-1).numpy(), bins=255)
    plt.show()
    '''


if __name__ == "__main__":
    main(sys.argv[1:])
