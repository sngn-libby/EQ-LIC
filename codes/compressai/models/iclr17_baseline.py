import os
import torch
import torch.nn as nn
from .GDN import GDN

import math
from .bitEstimator import BitEstimator


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class Analysis_net_17(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):
        super(Analysis_net_17, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.conv3(x)
        return x


class Synthesis_net_17(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192):
        super(Synthesis_net_17, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, 3, 9, stride=4, padding=4, output_padding=3)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x


class ICLR17Baseline(nn.Module):
    def __init__(self, out_channel_N=128):
        super(ICLR17Baseline, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N

    def forward(self, x):
        quant_noise_feature = torch.zeros(x.size(0), self.out_channel_N, x.size(2) // 16,
                                          x.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)

        y = self.Encoder(x)
        if self.training:
            y_hat = y + quant_noise_feature
        else:
            y_hat = torch.round(y)
        x_hat = self.Decoder(y_hat)
        y_likelihoods = self.bitEstimator(y_hat + 0.5) - self.bitEstimator(y_hat - 0.5)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }
