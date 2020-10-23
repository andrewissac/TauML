# region ######### Classes and Functions ######### 
from enum import Enum, unique
# If needed to compare Enum with int -> use IntEnum
# Category.name -> e.g. Category.GenuineTau.name == 'GenuineTau'
# Category.value -> e.g. Category.GenuineTau.value == 0.0
@unique
class MLCategory(Enum):
    GenuineTau = 0.0
    FakeTau = 1.0

@unique
class TBranches(Enum): # these are branches in TTrees, also used name "TBranches", to not collide with ROOTs TBranch class
    Tau_pt = 0.0
    Tau_eta = 1.0
    Tau_phi = 2.0
    Tau_mass = 3.0
    Tau_dxy = 4.0
    Tau_decayMode = 5.0
    Tau_ecalEnergy = 6.0
    Tau_hcalEnergy = 7.0
    Tau_ip3d = 8.0

# NEEDS TO BE MOVED TO UTILS OR SOMETHING
import numpy as np
def mapOneHotVectorToMLCategoryEnum(oneHotVector):
    oneHotMatrix = np.identity(len(MLCategory.__members__.items()))
    matchedEnumValue = None
    for i in range(oneHotMatrix.shape[0]):
        if(np.array_equal(oneHotMatrix[i], oneHotVector)):
            matchedEnumValue = float(i)
            break
    if matchedEnumValue is not None:
        return MLCategory(matchedEnumValue)
    else:
        return None

def mapMLCategoryEnumToOneHotVector(mlCategoryEnum):
    oneHotVector = np.zeros(len(MLCategory.__members__.items()))
    oneHotVector[int(mlCategoryEnum.value)] = 1.0
    return oneHotVector
# endregion ######### Classes ######### 

# region ######### Global stuff ######### 
# region ######### Paths and files ######### 
from os import path
basepath = '/work/aissac/'
testData_basepath = path.join(basepath, 'TestRootFiles')

fileDic = {
    MLCategory.GenuineTau : ['testDYJetsToLL.root'],
    MLCategory.FakeTau : ['testWJetsToLNu.root']
}
# endregion ######### Paths and files ######### 

# region ######### ML Lists ######### 
# get names (e.g. for a dict) by using: ml_variables[myIndex].name
# get value by using: ml_variables[myIndex].value
# ml_variables are values stored in branches from TTree
ml_variables = [member for name, member in TBranches.__members__.items()]
ml_categories = [member for name, member in MLCategory.__members__.items()]
# endregion ######### ML Lists ######### 
# endregion ######### Global stuff ######### 

