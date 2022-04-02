import torch as t

from .lsq import LsqQuan


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_ste(x):
    y = x.round()
    y_grad = 512 * (x - x.round()).pow(10)
    return (y - y_grad).detach() + y_grad


class STELsq(LsqQuan):
    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)  # s와 같은데 grad scale 적용된 버전
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_ste(x)
        x = x * s_scale
        return x
