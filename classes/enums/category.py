from enum import unique
from .enumbase import FloatEnumBase, EnumVal
# If needed to compare Enum with int -> use IntEnum
# Category.name -> e.g. Category.GenuineTau.name == 'GenuineTau'
# Category.value -> e.g. Category.GenuineTau.value == 0.0

@unique
class Category(FloatEnumBase):
    GenuineTau = EnumVal(0.0, 'Genuine Taus')
    FakeTau = EnumVal(1.0, 'Fake Taus')