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


class LiqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        self.bit = bit
        self.qmax = 2 ** (bit-1) - 1
        self.per_channel = per_channel
        self.interval = t.nn.Parameter(t.ones(1))

    def init_weight(self, x, *args, **kwargs):
        if self.per_channel:
            self.interval = t.nn.Parameter(
                x.detach().abs().mean(dim=(1, 2, 3), keepdim=True))
        else:
            self.interval = t.nn.Parameter(x.detach().abs().mean())

    def init_from_batch(self, x, *args, **kwargs):
        if self.per_channel:
            self.interval = t.nn.Parameter(
                x.detach().abs().mean(dim=(0, 2, 3), keepdim=True))
        else:
            self.interval = t.nn.Parameter(x.detach().abs().mean())

    def forward(self, x):
        x = x / self.interval
        x = t.clamp(x, -1, 1)
        x = x * self.qmax
        x = round_pass(x)
        x = x / self.qmax
        x = x * self.interval
        return x
