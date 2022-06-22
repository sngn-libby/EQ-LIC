import torch
import torch.nn as nn

from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from . import MeanScaleHyperprior
from .utils import conv, deconv


class ResBottleneck(nn.Module):
    def __init__(self, N):
        super(ResBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(N, N//2, 1),
            nn.ReLU(),
            nn.Conv2d(N//2, N//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(N//2, N, 1),
        )

    def forward(self, x):
        res = self.bottleneck(x)
        return x + res


class ResBlocks(nn.Sequential):
    def __init__(self, N, res_blocks):
        body = []
        for i in range(res_blocks):
            body.append(ResBottleneck(N))
        super(ResBlocks, self).__init__(*body)


class ResBlockMS(MeanScaleHyperprior):
    def __init__(self, N, M, res_blocks, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBlocks(N, res_blocks),
            conv(N, N),
            ResBlocks(N, res_blocks),
            conv(N, N),
            ResBlocks(N, res_blocks),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBlocks(N, res_blocks),
            deconv(N, N),
            ResBlocks(N, res_blocks),
            deconv(N, N),
            ResBlocks(N, res_blocks),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
