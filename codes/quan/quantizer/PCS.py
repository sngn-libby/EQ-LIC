import torch

from .quantizer import Quantizer


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class PCSWeight(Quantizer):
    def __init__(self, bit, **kwargs):
