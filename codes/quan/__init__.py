from .func import *
from .utils import *
import torch

global quant_loss
quant_loss = torch.zeros(1).squeeze()
