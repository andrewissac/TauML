from enum import Enum, unique
from .enumbase import EnumBase
# If needed to compare Enum with int -> use IntEnum
# Category.name -> e.g. Category.GenuineTau.name == 'GenuineTau'
# Category.value -> e.g. Category.GenuineTau.value == 0.0

@unique
class Category(EnumBase):
    GenuineTau = 0.0
    FakeTau = 1.0