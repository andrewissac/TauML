from enum import unique
from .enumbase import IntEnumBase

@unique
class Optimizer(IntEnumBase):
    Adadelta = 0
    Adagrad = 1
    Adam = 2
    Adamax = 3
    Ftrl = 4
    Nadam = 5
    RMSprop = 6
    SGD = 7
    