import torch
from torchvision import transforms
from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader

from compressai.zoo import models

from utils import config
from quan.utils import find_modules_to_quantize, replace_module_by_names

import examples.train as train

import os

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
    state_dict = torch.load('../experiments/Factorized_fp_q5/checkpoints/checkpoint_006.pth.tar')
    model = models["bmshj2018-factorized"](quality=1, pretrained=True)

    quant_config = config.get_config('configs/quant_config.yaml')
    lsq_model = models["bmshj2018-factorized"](quality=1)
    lsq_model.load_state_dict(model.state_dict())
    modules_to_replace = find_modules_to_quantize(lsq_model, quant_config.quan)
    replace_module_by_names(lsq_model, modules_to_replace)

    model.eval()
    lsq_model.eval()
    device = next(model.parameters()).device

    save_dir = os.path.join('../tests', 'pretrained_base_128bit')
    with torch.no_grad():
        d = next(iter(test_dataloader))
        d = d.to(device)
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
