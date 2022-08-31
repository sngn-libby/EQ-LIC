import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from . import MeanScaleHyperprior

from .utils import conv, deconv, update_registered_buffers


class MSReLU(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(),
            conv(N, N),
            nn.ReLU(),
            conv(N, N),
            nn.ReLU(),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(),
            conv(N, N),
            nn.ReLU(),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.ReLU(),
            deconv(M, M * 3 // 2),
            nn.ReLU(),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


class MSReLU6(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU6(),
            conv(N, N),
            nn.ReLU6(),
            conv(N, N),
            nn.ReLU6(),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU6(),
            deconv(N, N),
            nn.ReLU6(),
            deconv(N, N),
            nn.ReLU6(),
            deconv(N, 3),
        )


class MSTanh(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.Tanh(),
            conv(N, N),
            nn.Tanh(),
            conv(N, N),
            nn.Tanh(),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.Tanh(),
            deconv(N, N),
            nn.Tanh(),
            deconv(N, N),
            nn.Tanh(),
            deconv(N, 3),
        )

