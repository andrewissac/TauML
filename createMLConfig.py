import classes.mlconfig as cfg
import datetime
from classes.mlparameters import MLParameterset
from classes.enums.category import Category
from classes.enums.tbranches import TBranches
from os import path

datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MLCONFIG = cfg.MLConfig()
MLCONFIG.baseWorkPath = '/work/aissac/'
MLCONFIG.baseCephPath = '/ceph/aissac/'

# get names (e.g. for a dict) by using: variables[myIndex].name
# get value by using: variables[myIndex].value
# variables are values stored in branches from TTree
MLCONFIG.variables = [member for name, member in TBranches.getAllMembers()]
MLCONFIG.categories = [member for name, member in Category.getAllMembers()]
MLCONFIG.encodeLabels_OneHot = True
MLCONFIG.generateHistograms = True

# region small test dataset
MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseWorkPath, 'TestRootFiles')
MLCONFIG.outputPath = path.join(MLCONFIG.baseWorkPath, 'TauAnalyzer/ML/output_smallDataset_{}'.format(datetimeStr))
MLCONFIG.plotsOutputPath = path.join(MLCONFIG.outputPath, "plots")

MLCONFIG.plotsOutputPath = path.join(MLCONFIG.outputPath, "plots")
from pathlib import Path
Path(MLCONFIG.outputPath).mkdir(parents=True, exist_ok=True)
Path(MLCONFIG.plotsOutputPath).mkdir(parents=True, exist_ok=True)

MLCONFIG.datasetsList = [
    (Category.GenuineTau, [path.join(MLCONFIG.datasetsBasePath, 'GenuineTau')]), 
    (Category.FakeTau, [path.join(MLCONFIG.datasetsBasePath, 'FakeTau')])
    ]
MLCONFIG.saveToJsonfile(MLCONFIG.outputPath, 'cfg.json')
# endregion small test dataset

# # region large test dataset
# MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseCephPath, 'mergedROOTFiles')
# MLCONFIG.outputPath = path.join(MLCONFIG.baseWorkPath, 'TauAnalyzer/ML/output_wholeDataset_{}'.format(datetimeStr))

# MLCONFIG.plotsOutputPath = path.join(MLCONFIG.outputPath, "plots")
# from pathlib import Path
# Path(MLCONFIG.outputPath).mkdir(parents=True, exist_ok=True)
# Path(MLCONFIG.plotsOutputPath).mkdir(parents=True, exist_ok=True)

# MLCONFIG.datasetsList = [
#     (Category.GenuineTau, [
#         path.join(MLCONFIG.datasetsBasePath, 'output__DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1_MINIAODSIM')
#         ]), 
#     (Category.FakeTau,[
#         path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14_ext1-v2_MINIAODSIM'),
#         path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3_MINIAODSIM')
#         ]) 
#     ]
# endregion large test dataset

MLCONFIG.mlparametersetPath = '/work/aissac/TauAnalyzer/ML/parametersets/mlparams0001.json'
MLCONFIG.mlparams = MLParameterset.loadFromJsonfile(MLCONFIG.mlparametersetPath)

cfgOutputFileName = 'cfg.json'
cfgFile = Path(path.join(MLCONFIG.outputPath, cfgOutputFileName))
if cfgFile.is_file():
    userInput = input('Config file already exists! Do you want to overwrite the config file? (y/n)\n')
    userInput = userInput.lower()
    if userInput == 'y':
        MLCONFIG.saveToJsonfile(MLCONFIG.outputPath, cfgOutputFileName)
        print("Config file overwritten.")
    else:
        print('Config file was not overwritten.')
        pass

