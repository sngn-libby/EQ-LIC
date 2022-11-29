import torch
from compressai.zoo import models
from quan.utils import find_modules_to_quantize, replace_module_by_names
from utils import util, config
from thop import profile

model = models['gmm-relu'](quality=1)
# quant_config = config.get_config('configs/joint_8bit_pams.yaml')
# modules_to_replace = find_modules_to_quantize(model, quant_config.quan)
# model = replace_module_by_names(model, modules_to_replace)

param_size = 0
for name, param in model.named_parameters():
    if name.endswith('alpha') or name.endswith('fn.s'):
        param_size += param.nelement() * param.element_size()
    else:
        param_size += param.nelement() * param.element_size() / 4
print('param size: {:.3f}MB'.format(param_size / 1024**2))
buffer_size = 0
for name, buffer in model.named_buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

input = torch.randn(1, 3, 256, 256)
macs, params = profile(model, inputs=(input, ))
print(macs)
print(params)