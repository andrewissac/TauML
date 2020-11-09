from enum import unique
from .enumbase import IntEnumBase

@unique
class Activation(IntEnumBase):
    elu = 0
    exponential = 1
    hard_sigmoid = 2
    linear = 3
    relu = 4
    selu = 5
    sigmoid = 6
    softmax = 7
    softplus = 8
    softsign = 9
    swish = 10
    tanh = 11