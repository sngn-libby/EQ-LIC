import torch as t
import quan
import math
from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQelWeight(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.FloatTensor([1]).squeeze() / (self.thd_pos ** 0.5))

    def init_weight(self, x, *args, **kwargs):
        if self.per_channel:
            self.s.data = x.detach().abs().amax(dim=(1, 2, 3), keepdim=True) / self.thd_pos
        else:
            self.s.data = x.detach().abs().amax() / self.thd_pos

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)  # s와 같은데 grad scale 적용된 버전

        x_hat = x / s_scale
        x_hat = t.clamp(x_hat, self.thd_neg, self.thd_pos)
        x_hat = round_pass(x_hat)
        x_hat = x_hat * s_scale

        if self.training:
            # quan.quant_loss = quan.quant_loss.to(x.device)
            # quan.quant_loss += ((x_hat - x).abs()).sum() / len(x.reshape(-1))
            m = 2
            quan.quant_loss = quan.quant_loss.to(x.device)
            quan.quant_loss += ((x_hat - x)**m).sum()**(1/m) / len(x.reshape(-1))
        return x_hat


class LsqQelAct(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        self.inited = False
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.FloatTensor([1]).squeeze() / self.thd_pos)

    def init_activation(self, x, *args, **kwargs):
        self.s.data = x.detach().abs().amax() / self.thd_pos
        if self.per_channel:
            self.s.data = self.s.data.detach().expand(x.shape[1]).clone().unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()

    def forward(self, x):
        if not self.inited and self.training:
            self.inited = True
            self.init_activation(x)

        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)  # s와 같은데 grad scale 적용된 버전

        x_hat = x / s_scale
        x_hat = t.clamp(x_hat, self.thd_neg, self.thd_pos)
        x_hat = round_pass(x_hat)
        x_hat = x_hat * s_scale

        if self.training:
            m = 2
            quan.quant_loss = quan.quant_loss.to(x.device)
            quan.quant_loss += ((x_hat - x)**m).sum()**(1/m) / len(x.reshape(-1))
            # quan.quant_loss = quan.quant_loss.to(x.device)
            # quan.quant_loss += ((x_hat - x).abs()).sum() / len(x.reshape(-1))
        return x_hat
