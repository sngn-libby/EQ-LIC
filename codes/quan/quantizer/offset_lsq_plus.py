import torch as t

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class OffsetLSQPlus(Quantizer):
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
        self.s = t.nn.Parameter(t.ones(1) / (self.thd_pos ** 0.5))
        self.b = t.nn.Parameter(t.tensor([float(-1e-9)]))
        self.batch_init = 20
        self.init_state = 0

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=(1, 2, 3), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            self.b = t.nn.Parameter(t.ones(x.shape[0], 1, 1, 1))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def init_from_batch(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=(0, 2, 3), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            self.b = t.nn.Parameter(t.ones(1, x.shape[1], 1, 1)) * -1e-9
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.init_state == 0:
            mina = t.min(x.detach())
            self.s.data = (t.max(x.detach()) - mina) / (self.thd_pos - self.thd_neg)
            self.b.data = mina - self.s.data * self.thd_neg
            self.init_state += 1
        elif self.init_state < self.batch_init:
            mina = t.min(x.detach())
            self.s.data = self.s.data * 0.9 + 0.1 * (t.max(x.detach()) - mina) / (self.thd_pos - self.thd_neg)
            self.b.data = self.s.data * 0.9 + 0.1 * (mina - self.s.data * self.thd_neg)
            self.init_state += 1

        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)  # s와 같은데 grad scale 적용된 버전
        x = (x - self.b) / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)   # bar x
        x = x * s_scale + self.b  # x hat
        return x
