from enum import Enum, EnumMeta
from numpy import zeros

# Enables using the first value of an Enum Class to be the default value by default constructor MyEnum()
class DefaultEnumMeta(EnumMeta):
    default = object()
    def __call__(cls, value=default, *args, **kwargs):
        if value is DefaultEnumMeta.default:
            # Assume the first enum is default
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)

class EnumBase(Enum, metaclass=DefaultEnumMeta):
    def __int__(self):
        return int(self.value)

    def __len__(self):
        return len(self.getAllMembers())

    @classmethod
    def getAllMembers(cls):
        return cls.__members__.items()
    
    @classmethod
    def getAllValues(cls):
        return [m.value for m in cls]

    @classmethod
    def getAllNames(cls):
        return [m.name for m in cls]

    @classmethod
    def oneHotVectorToEnum(cls, oneHotVector):
        """
        Usage example for class Category(EnumBase):
        import numpy as np
        oneHotVector = np.array([0., 1.])
        print(Category.oneHotVectorToEnum(oneHotVector))
        """
        allEnumMember = cls.getAllMembers()
        if(len(oneHotVector) != len(allEnumMember)):
            raise Exception("Dimension of the one-hot-encoded vector does not match with the specified enum.")

        from numpy import identity, array_equal
        oneHotMatrix = identity(len(allEnumMember))
        matchedEnumValue = None
        for i in range(oneHotMatrix.shape[0]):
            if(array_equal(oneHotMatrix[i], oneHotVector)):
                matchedEnumValue = float(i)
                break
        if matchedEnumValue is not None:
            return cls(matchedEnumValue)
        else:
            return None

    def toOneHotVector(self):
        """
        Usage example for class Category(EnumBase):
        myCategory = Category.FakeTau
        myOneHotVector= myCategory.toOneHotVector()
        """
        oneHotVector = zeros(len(self))
        oneHotVector[int(self)] = 1.0
        return oneHotVector

    # maybe useful one day..
    # def __new__(cls, *args, **kwargs):
    #     obj = object.__new__(cls)
    #     obj._value_ = args
    #     return obj

    # # if more than a tuple with 2 values is needed for an enum -> override it 
    # def __init__(self, value: float, displayname: str = None):
    #     self._firstvalue_ = value
    #     self._displayname_ = displayname

    # @property
    # def firstvalue(self):
    #     return self._firstvalue_

    # @property
    # def displayname(self):
    #     return self._displayname_