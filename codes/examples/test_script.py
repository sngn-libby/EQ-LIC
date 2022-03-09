import torch
from torchvision import transforms
from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader

from compressai.zoo import models

from utils import config
from quan.utils import find_modules_to_quantize, replace_module_by_names

import examples.train as train

import os


def print_layer_weights(weight1, weight2):
    out_ch, in_ch, h, w = weight1.shape
    n = 1
    for i in range(out_ch):
        print(f'---{i}th output channel---')
        for j in range(0, in_ch, n):
            if j == in_ch - in_ch % n and in_ch % n != 0:
                continue
            print(f'{j} ~ {j + n-1} input channel\n')
            for k in range(h):
                for s in range(w * n):
                    print(f'{weight1[i][j + s // w][k][s % w]:.3f}', end=' ')
                    if s + 1 == n * w:
                        print(end='\n')
                    elif (s + 1) % w == 0:
                        print(end='\t')
            print()
    exit()


def print_feature_channel_variance(fea1, fea2):
    var1 = fea1.var(dim=(2, 3))
    var2 = fea2.var(dim=(2, 3))
    print(var1.shape)
    for i in range(var1.shape[1]):
        print(f'channel {i}, var1 = {var1[0][i]:.8f}')
        print(f'channel {i}, var2 = {var2[0][i]:.8f}')
    for i in range(var1.shape[1]):
        print(f'channel {i} proportion: {var1[0][i]/var2[0][i]:.8f}')
    exit()


if __name__ == '__main__':
    test_transforms = transforms.Compose([transforms.CenterCrop(256),
                                          transforms.ToTensor()])
    test_dataset = ImageFolder('c:/flicker', split="test", transform=test_transforms)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 shuffle=False,
                                 pin_memory=True,
                                 )

    # create models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models["mbt2018-mean"](quality=5, pretrained=True)

    lsq_model = models["mbt2018-mean"](quality=5, pretrained=True)
    # lsq_model.load_state_dict(model.state_dict())
    quant_config = config.get_config('configs/quant_config.yaml')
    modules_to_replace = find_modules_to_quantize(lsq_model, quant_config.quan)
    replace_module_by_names(lsq_model, modules_to_replace)

    model.to(device)
    lsq_model.to(device)

    model.eval()
    lsq_model.eval()

    # print_layer_weights(model.g_s[6].weight, lsq_model.g_s[6].quan_w_fn(lsq_model.g_s[6].weight))

    save_dir = os.path.join('../tests', 'mean-scale-init-test')
    with torch.no_grad():
        d = next(iter(test_dataloader))
        d = d.to(device)

        '''print hidden feature'''
        # y1 = model.g_a[0:7](d)
        # y2 = lsq_model.g_a[0:7](d)
        # y1_hat, _ = model.entropy_bottleneck(y1)
        # y2_hat, _ = lsq_model.entropy_bottleneck(y2)
        # fea1 = model.g_s[0:7](y1_hat)
        # fea2 = lsq_model.g_s[0:7](y2_hat)
        # print_feature_channel_variance(fea1, fea2)

        '''replace one layer'''
        # x = model.g_a[0:6](d)
        # y = lsq_model.g_a[6](x)
        # y = model.g_a(d)
        # y_hat, _ = model.entropy_bottleneck(y)
        # x_hat = model.g_s[0:6](y_hat)
        # x_hat = lsq_model.g_s[6](x_hat)
        # # x_hat = model.g_s[1:7](x_hat)
        # x_hat = train.torch2img(x_hat[0])
        # x_hat.show()
        # exit()

        '''init activation LSQ from sample activation'''
        train.init_act_lsq(lsq_model, test_dataloader)

        out = model(d)
        lsq_out = lsq_model(d)

        rec = train.torch2img(out['x_hat'][0])
        lsq_rec = train.torch2img(lsq_out['x_hat'][0])
        img = train.torch2img(d[0])

        lsq_rec.show()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        rec.save(os.path.join(save_dir, 'rec.png'))
        lsq_rec.save(os.path.join(save_dir, 'rec_lsq.png'))
        img.save(os.path.join(save_dir, 'gt.png'))
