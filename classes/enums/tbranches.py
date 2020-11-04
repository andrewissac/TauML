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

    @classmethod
    def getDisplaynames(self):
        return {
            TBranches.Tau_pt: 'Tau pt',
            TBranches.Tau_eta: 'Tau eta',
            TBranches.Tau_phi: 'Tau phi',
            TBranches.Tau_mass: 'Tau mass',
            TBranches.Tau_dxy: 'Tau dxy',
            TBranches.Tau_decayMode: 'Tau decaymode',
            TBranches.Tau_ecalEnergy: 'Tau ecal energy',
            TBranches.Tau_hcalEnergy: 'Tau hcal energy',
            TBranches.Tau_ip3d: 'Tau ip3d'
            }
            
    def getDisplayname(self):
        return self.getDisplaynames()[self]

