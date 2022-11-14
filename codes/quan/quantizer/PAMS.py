import torch

from .quantizer import Quantizer


def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-10


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class PAMSWeight(Quantizer):
    """
    Quantization function for quantize weight with maximum.
    """
    def __init__(self, bit, **kwargs):
        super().__init__(bit)
        self.k_bits = bit
        self.qmax = 2. ** (bit - 1) - 1.

    def forward(self, input):
        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = round_pass(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight


class PAMSAct(Quantizer):
    """
    Quantization function for quantize activation with parameterized max scale.
    """

    def __init__(self, bit, ema_epoch=0, decay=0.9997, **kwargs):
        super().__init__(bit)
        self.decay = decay
        self.k_bits = bit
        self.qmax = 2. ** self.k_bits - 1.
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.ema_epoch = ema_epoch
        self.epoch = 0
        self.register_buffer('max_val', torch.ones(1).squeeze(0))
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.constant_(self.alpha, 10)

    def _ema(self, x):
        max_val = x.amax(dim=(1, 2, 3)).mean()
        # mean from each data batch [C, W, H]'s absolute max value
        if self.epoch == 0:
            self.max_val = max_val
        else:
            self.max_val = (1.0 - self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):
        if self.epoch > self.ema_epoch or not self.training:
            act = torch.clamp(x, -self.alpha, self.alpha)

        elif self.epoch <= self.ema_epoch and self.training:
            act = x
            self._ema(x)
            self.alpha.data = self.max_val.unsqueeze(0)
            print('my name is doof and you do what i say')

        act = act * self.qmax / self.alpha
        q_act = round_pass(act)
        q_act = q_act * self.alpha / self.qmax

        return q_act