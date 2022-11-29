import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from compressai.zoo.image import model_architectures as architectures
from compressai.zoo import models
from quan.utils import find_modules_to_quantize, replace_module_by_names
from utils import util, config
from examples.train import RateDistortionLoss
import quan


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LinearQ(torch.nn.Module):
    def __init__(self, bit):
        super().__init__()
        self.epoch = 0
        # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
        self.thd_neg = 0
        self.thd_pos = 2 ** bit - 1
        self.s = torch.nn.Parameter(torch.FloatTensor([0.05]).squeeze()) # / self.thd_pos

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s = grad_scale(self.s, s_grad_scale)
        s = self.s
        x = x / s * self.thd_pos
        x = torch.clamp(x, 0, self.thd_pos)
        x = round_pass(x)
        x = x * s / self.thd_pos
        return x


def main():
    train_transforms = transforms.Compose(
        [transforms.RandomCrop((256, 256)), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
    )

    train_dataset = ImageFolder('c:/flicker', split="train", transform=train_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = models['ms-relu'](quality=4)
    quant_config = config.get_config('configs/ms_lsqqel.yaml')
    modules_to_replace = find_modules_to_quantize(net, quant_config.quan)
    net = replace_module_by_names(net, modules_to_replace)

    checkpoint = torch.load('../experiments/MS_lsq_q4/checkpoints/checkpoint_best_loss.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])

    net = net.to(device)

    parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles")
    ]
    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=1e-4,
        weight_decay=0,
    )
    MSELoss = torch.nn.MSELoss()

    net.train()
    for i, d in enumerate(train_dataloader):
        net.apply(lambda m: setattr(m, 'inited', True))
        d = d.to(device)
        optimizer.zero_grad()

        y = net.g_a(d)
        z = net.h_a(y)
        z_hat, _ = net.entropy_bottleneck(z)
        gaussian_params = net.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, _ = net.gaussian_conditional(y, scales_hat, means=means_hat)

        a = net.g_s[0:4](y_hat)
        a.retain_grad()

        layer = net.g_s[4]
        s = layer.quan_a_fn.s
        a_hat = layer.quan_a_fn(a)
        a_hat.retain_grad()
        w_hat = layer.quan_w_fn(layer.weight)
        x_hat = torch.nn.functional.conv_transpose2d(a_hat, w_hat, layer.bias, stride=layer.stride,
                                                     padding=layer.padding, output_padding=layer.output_padding,
                                                     dilation=layer.dilation)
        x_hat = net.g_s[5:7](x_hat)

        # loss = MSELoss(x_hat, d)
        # loss.backward()
        # print('loss = ', loss)
        quan.quant_loss.backward()

        print('내가 이론적으로 구한 s의 gradient 식이 맞는가?')
        grad_pred = a_hat.grad * a_hat/s - a_hat.grad * a/s * (a.grad != 0)
        grad_pred = grad_pred.sum()
        print('s.grad = ', s.grad)
        print('grad_pred = ', grad_pred)
        print('grad_pred/s.grad = ', grad_pred/s.grad)

        print('clamp 구간 내부 개수: ', (a <= s * 255).sum())
        print('clamp 구간 외부 개수: ', (a > s * 255).sum())

        print('clamp 구간 내부 gradient의 방향과 합')
        grad_inter = (a_hat.grad * a_hat/s - a_hat.grad * a/s) * (a <= s * 255)
        print('how many grad_inter > 0 :', (grad_inter > 0).sum())
        print(f'mean of grad_inter > 0 : {(grad_inter * (grad_inter > 0)).sum() / (grad_inter > 0).sum():.3g}', )
        print('how many grad_inter < 0 :', (grad_inter < 0).sum())
        print(f'mean of grad_inter < 0 : {(grad_inter * (grad_inter < 0)).sum() / (grad_inter < 0).sum():.3g}')
        print(f'sum of grad_inter = {grad_inter.sum():.3g}')

        print('clamp 구간 외부 gradient의 방향과 합')
        grad_outer = (a_hat.grad * a_hat / s) * (a > s * 255)
        print('how many grad_outer > 0 :', (grad_outer > 0).sum())
        print(f'sum of grad_outer > 0 : {(grad_outer * (grad_outer > 0)).sum():.3g}')
        print('how many grad_outer < 0 :', (grad_outer < 0).sum())
        print(f'sum of grad_outer < 0 : {(grad_outer * (grad_outer < 0)).sum():.3g}')
        print(f'sum of grad_outer = {grad_outer.sum():.3g}')

        print('grad_pred', grad_pred)
        print(f'grad_sum {grad_inter.sum() + grad_outer.sum():.3g}')

        break


if __name__ == "__main__":
    main()
