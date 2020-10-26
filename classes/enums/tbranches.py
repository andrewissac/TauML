from enum import Enum, unique
from .enumbase import EnumBase
# If needed to compare Enum with int -> use IntEnum
# TBranches.name -> e.g. TBranches.Tau_pt.name == 'Tau_pt'
# TBranches.value -> e.g. TBranches.Tau_pt.value == 0.0

@unique
class TBranches(EnumBase): # these are branches in TTrees, also used name "TBranches" to not collide with ROOTs TBranch class
    Tau_pt = 0.0
    Tau_eta = 1.0
    Tau_phi = 2.0
    Tau_mass = 3.0
    Tau_dxy = 4.0
    Tau_decayMode = 5.0
    Tau_ecalEnergy = 6.0
    Tau_hcalEnergy = 7.0
    Tau_ip3d = 8.0
