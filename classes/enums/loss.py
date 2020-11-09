from enum import unique
from .enumbase import IntEnumBase

@unique
class LossFunction(IntEnumBase):
    BinaryCrossentropy = 0
    CategoricalCrossentropy = 1
    CategoricalHinge = 2
    CosineSimilarity = 3
    Hinge = 4
    Huber = 5
    KLDivergence = 6
    LogCosh = 7
    MeanAbsoluteError = 8
    MeanAbsolutePercentageError = 9
    MeanSquaredError = 10
    MeanSquaredLogarithmicError = 11
    Poisson = 12
    SparseCategoricalCrossentropy = 13
    SquaredHinge = 14

