import torch
import torch.nn as nn

from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from . import ScaleHyperprior
from .utils import conv, deconv


class AuxMeanScale(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a_body = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )
        self.g_a_tail = nn.Sequential(
            GDN(N),
            conv(N, M)
        )

        self.g_a_aux = conv(3, N // 16)
        self.avg_pool = nn.AvgPool2d(7, padding=3, stride=4)
        self.max_pool = nn.MaxPool2d(7, padding=3, stride=4)
        self.pixel_unshuffle4 = nn.PixelUnshuffle(4)

        self.g_s_aux = deconv(M, 3*16)
        self.pixel_shuffle4 = nn.PixelShuffle(4)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )
        self.h_a_aux = conv(M, N, stride=4, kernel_size=5)
        self.pixel_unshuffle2 = nn.PixelUnshuffle(2)

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.h_s_aux = deconv(N, M * 2 * 4)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        y = self.g_a_body(x) # + self.pixel_unshuffle4(self.g_a_aux(x))
        y = self.g_a_tail(y)
        z = self.h_a(y) + self.h_a_aux(y) # self.pixel_unshuffle2(self.h_a_aux(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        bool = torch.any(z_likelihoods == 0)
        z_likelihoods = torch.nan_to_num(z_likelihoods)
        gaussian_params = self.h_s(z_hat) # + self.pixel_shuffle2(self.h_s_aux(z_hat))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        bool = torch.any(y_likelihoods == 0)
        y_likelihoods = torch.nan_to_num(y_likelihoods)
        x_hat = self.g_s(y_hat) # + self.pixel_shuffle4(self.g_s_aux(y_hat))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def init_act(self, dataloader):
        print('initializing act lsq')
        device = next(self.parameters()).device
        x = next(iter(dataloader)).to(device)

        # x = y
        for i in range(7):
            if i in [0, 2, 4, 6]:
                self.g_a[i].quan_a_fn.init_from_batch(x)
            x = self.g_a[i](x)
            print(i, torch.any(torch.isnan(x)))

        # for hyper-prior model, h_a
        z = x  # 여기서 x가 안 바뀌어야 할텐데...
        for i in range(5):
            if i in [0, 2, 4]:
                self.h_a[i].quan_a_fn.init_from_batch(z)
            z = self.h_a[i](z)
            print(i, torch.any(torch.isnan(z)))
        z = z + self.h_a_aux(x)
        z, likelihood = self.entropy_bottleneck(z)
        print('z hat', torch.any(torch.isnan(z)))
        print('likelihood', torch.any(torch.isnan(likelihood)))
        # x, _ = model.entropy_bottleneck(x)

        # for hyper-prior model, h_s
        for i in range(5):
            if i in [0, 2, 4]:
                self.h_s[i].quan_a_fn.init_from_batch(z)
            z = self.h_s[i](z)
            print(i, torch.any(torch.isnan(z)))

        # for hyper-prior model, gaussian_params, x = y_hat
        scales_hat, means_hat = z.chunk(2, 1)
        x, _ = self.gaussian_conditional(x, scales_hat, means=means_hat)

        # x = x_hat
        for i in range(7):
            if i in [0, 2, 4, 6]:
                self.g_s[i].quan_a_fn.init_from_batch(x)
            x = self.g_s[i](x)
            print(i, torch.any(torch.isnan(x)))

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
