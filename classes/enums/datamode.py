from enum import unique
from .enumbase import IntEnumBase
# If needed to compare Enum with int -> use IntEnum
# Category.name -> e.g. Category.GenuineTau.name == 'GenuineTau'
# Category.value -> e.g. Category.GenuineTau.value == 0.0

@unique
class Datamode(IntEnumBase):
    train = 0
    valid = 1
    test = 2