# region ######### Classes ######### 
from enum import Enum, unique
# If needed to compare Enum with int -> use IntEnum
# Category.name -> z.B. Category.GenuineTau.name == 'GenuineTau'
# Category.value -> z.B. Category.GenuineTau.value == 0.0
@unique
class MLCategory(Enum):
    GenuineTau = 0.0
    FakeTau = 1.0

@unique
class TBranches(Enum): # these are branches in TTrees
    Tau_pt = 0.0
    Tau_eta = 1.0
    Tau_phi = 2.0
    Tau_mass = 3.0
    Tau_dxy = 4.0
    Tau_decayMode = 5.0
    Tau_ecalEnergy = 6.0
    Tau_hcalEnergy = 7.0
    Tau_ip3d = 8.0
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

