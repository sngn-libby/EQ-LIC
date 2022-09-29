import torch
from .quantizer import Quantizer


def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class MICS(Quantizer):
    """
    Mean Interval Coefficient Scale
    """
    def __init__(self, bit, all_positive=False, ema_epoch=0, decay=0.9997, **kwargs):
        super().__init__(bit)
        self.decay = decay
        self.k_bits = bit
        self.all_positive = all_positive
        if all_positive:
            self.qmax = 2. ** self.k_bits - 1.
        else:
            self.qmax = 2. ** (self.k_bits - 1) - 1.
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.epoch = 0

    def forward(self, x):
        # var = torch.var(x.detach()) + 1e-8
        # if self.epoch == 0 and self.training:
        #     self.alpha.data = torch.max(x.detach().abs()) / abs_mean + 1e-8
        interval = torch.max(x.detach().abs()) * self.alpha + 1e-8
        if self.all_positive:
            x = torch.clamp(x, interval * 0, interval)
        else:
            x = torch.clamp(x, -interval, interval)

        x = x * self.qmax / interval
        q_x = round_pass(x)
        q_x = q_x * interval / self.qmax

        return q_x
