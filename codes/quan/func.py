import torch as t
import torch.nn.functional as F
from compressai.layers.gdn import GDN


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        self.bias = None
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight, self.bias)


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)


class QuanConvTranspose2d(t.nn.ConvTranspose2d):
    def __init__(self, m: t.nn.ConvTranspose2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.ConvTranspose2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         output_padding=m.output_padding,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        self.bias = None
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        assert isinstance(self.padding, tuple)
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return F.conv_transpose2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)

class QuanGDN(GDN):
    def __init__(self, m: GDN, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == GDN
        super().__init__(m.beta.shape[0], inverse=m.inverse)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

    def forward(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)

        if self.inverse:
            norm = t.sqrt(norm)
        else:
            norm = t.rsqrt(norm)

        out = x * norm

        return out



QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear,
    t.nn.ConvTranspose2d: QuanConvTranspose2d
    # GDN: QuanGDN,
}
