from enum import unique
from .enumbase import FloatEnumBase, EnumVal
# If needed to compare Enum with int -> use IntEnum
# TBranches.name -> e.g. TBranches.Tau_pt.name == 'Tau_pt'
# TBranches.value -> e.g. TBranches.Tau_pt.value == 0.0

@unique
class TBranches(FloatEnumBase): # these are branches in TTrees, also used name "TBranches" to not collide with ROOTs TBranch class
    Tau_pt = EnumVal(0.0, 'Tau pt')
    Tau_eta = EnumVal(1.0, 'Tau eta')
    Tau_phi = EnumVal(2.0, 'Tau phi')
    Tau_mass = EnumVal(3.0, 'Tau mass')
    Tau_dxy = EnumVal(4.0, 'Tau dxy')
    Tau_decayMode = EnumVal(5.0, 'Tau decaymode')
    Tau_ecalEnergy = EnumVal(6.0, 'Tau ecal energy')
    Tau_hcalEnergy = EnumVal(7.0, 'Tau hcal energy')
    Tau_ip3d = EnumVal(8.0, 'Tau ip3d')

