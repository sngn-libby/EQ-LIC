import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import ResidualBlockReLU, ResidualBlockUpsample, ResidualBlockUpsampleReLU, ResidualBlockWithStrideReLU, conv3x3, subpel_conv3x3
from . import MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors, Cheng2020Anchor

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


class JointReLU(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )


class GMMReLU(Cheng2020Anchor):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStrideReLU(3, N, stride=2),
            ResidualBlockReLU(N, N),
            ResidualBlockWithStrideReLU(N, N, stride=2),
            ResidualBlockReLU(N, N),
            ResidualBlockWithStrideReLU(N, N, stride=2),
            ResidualBlockReLU(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.ReLU(),
            conv3x3(N, N),
            nn.ReLU(),
            conv3x3(N, N, stride=2),
            nn.ReLU(),
            conv3x3(N, N),
            nn.ReLU(),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.ReLU(),
            subpel_conv3x3(N, N, 2),
            nn.ReLU(),
            conv3x3(N, N * 3 // 2),
            nn.ReLU(),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.ReLU(),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
        )

        self.g_s = nn.Sequential(
            ResidualBlockReLU(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockReLU(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockReLU(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockReLU(N, N),
            subpel_conv3x3(N, 3, 2),
        )
