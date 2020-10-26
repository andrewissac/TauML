from enum import Enum

def OneHotVectorToEnum(oneHotVector, EnumClass: Enum) -> Enum:
    allEnumMembers = EnumClass.__members__.items()
    if(len(oneHotVector) != len(allEnumMembers)):
        raise Exception("Dimension of the one-hot-encoded vector does not match with the specified enum.")

    from numpy import identity, array_equal
    oneHotMatrix = identity(len(allEnumMembers))
    matchedEnumValue = None
    for i in range(oneHotMatrix.shape[0]):
        if(array_equal(oneHotMatrix[i], oneHotVector)):
            matchedEnumValue = float(i)
            break
    if matchedEnumValue is not None:
        return EnumClass(matchedEnumValue)
    else:
        return None


# import json
# from classes.enums.CategoryEnum import Category
# from classes.enums.TBranchesEnum import TBranches

# class EnumEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if type(obj) in Category.values():
#             return { "__MLClassEnum__" : str(obj)}
#         elif type(obj) in TBranches.values():
#             return {"__TBranchesEnum__" : str(obj)}
#         return json.JSONEncoder.default(self, obj)

# def as_enum(dic):
#     if "__CategoryEnum__" in dic:
#         name, member = dic["__CategoryEnum__"].split(".")
#         return getattr(Category[name], member)
#     elif "__TBranchesEnum__" in dic:
#         name, member = dic["__TBranchesEnum__"].split(".")
#         return getattr(TBranches[name], member)
#     else:
#         return dic