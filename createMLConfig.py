import classes.mlconfig as cfg
from classes.enums.category import Category
from classes.enums.tbranches import TBranches
from os import path

MLCONFIG = cfg.MLConfig()
MLCONFIG.baseWorkPath = '/work/aissac/'
MLCONFIG.baseCephPath = '/ceph/aissac/'
#MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseWorkPath, 'TestRootFiles')
MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseCephPath, 'mergedROOTFiles')
#MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_dxy-999_decayMode5_6_PrunedInAnalyzer")
#MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_dxy-999_decayMode5_6_PrunedInPython")
#MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_smallDataset")
MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_all")
MLCONFIG.datasetsList = [
    # (Category.GenuineTau, [path.join(MLCONFIG.datasetsBasePath, 'GenuineTau')]), 
    # (Category.FakeTau, [path.join(MLCONFIG.datasetsBasePath, 'FakeTau')])
    (Category.GenuineTau, [
        path.join(MLCONFIG.datasetsBasePath, 'output__DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1_MINIAODSIM')
        ]), 
    (Category.FakeTau,[
        path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14_ext1-v2_MINIAODSIM'),
        path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3_MINIAODSIM')
        ]) 
    ]

# get names (e.g. for a dict) by using: variables[myIndex].name
# get value by using: variables[myIndex].value
# variables are values stored in branches from TTree
MLCONFIG.variables = [member for name, member in TBranches.getAllMembers()]
MLCONFIG.categories = [member for name, member in Category.getAllMembers()]


MLCONFIG.saveToJsonfile('/work/aissac/TauAnalyzer/ML', 'cfgWithWholeDataset.json')

