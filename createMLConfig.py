import datetime
import glob
from os import path
from classes.mlconfig import MLConfig
from classes.mlparameters import MLParameterset
from classes.enums.category import Category
from classes.enums.tbranches import TBranches
from classes.enums.datamode import Datamode
from classes.datasetinfo import DatasetInfo, DatasetInfoSummary, ROOTFileInfo
from utils.progressbar import progress, repeatList
from utils.rootfile_util import rootTTree2numpy

def getDatasetsInfoSummary(cfg: MLConfig):
    if not isinstance(cfg, MLConfig):
        raise TypeError

    #region count all files to display progress
    count = 0
    totalFileCount = 0
    tableflip = '(ノಠ 益ಠ)ノ彡┻━┻........'
    progressbarSegments = 60
    chars = [char for char in tableflip]
    from collections import deque
    cycleCharsDeque = deque(repeatList(chars, progressbarSegments))
    for category, datasets in cfg.datasetsList: 
        for datasetPath in datasets:
            temp = []
            for datamode in Datamode.getAllMembers():
                temp.append((datamode, [f for f in glob.glob(path.join(datasetPath, datamode.name, "*.root"), recursive=False)]))
            for datamode, filePaths in temp:
                for rootFile in filePaths:
                    totalFileCount += 1
    #endregion

    branchNames = [branch.name for branch in cfg.variables]
    datasetInfoList = []

    for category, datasets in cfg.datasetsList: 
        for datasetPath in datasets:
            fileInfos = []
            files = []
            for datamode in Datamode.getAllMembers():
                files.append((datamode, [f for f in glob.glob(path.join(datasetPath, datamode.name, "*.root"), recursive=False)]))

            for datamode, filePaths in files:
                for rootFile in filePaths:
                    try:
                        tree = rootTTree2numpy(rootFile)
                        eventCount = len(tree[branchNames[0]])
                        rootfileInfo = ROOTFileInfo(rootFile, datamode, hadErrorOnOpening=False, eventCount=eventCount)
                        fileInfos.append(rootfileInfo)
                        del tree
                    except KeyboardInterrupt:
                        return
                    except:
                        rootfileInfo = ROOTFileInfo(rootFile, datamode, hadErrorOnOpening=True, eventCount=None)
                        fileInfos.append(rootfileInfo)
                    # following section is only for progressbar depiction
                    progress(
                        count=count, total=totalFileCount, cycleCharsDeque=cycleCharsDeque, 
                        message='soon TM', progressbarSegments=progressbarSegments)
                    if(count % 2 == 0):
                        cycleCharsDeque.rotate(1)
                    count += 1

            datasetInfoList.append(DatasetInfo(category, datasetPath, fileInfos))
    return DatasetInfoSummary(datasetInfoList)

datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MLCONFIG = MLConfig()
MLCONFIG.baseWorkPath = '/work/aissac/'
MLCONFIG.baseCephPath = '/ceph/aissac/'

# get names (e.g. for a dict) by using: variables[myIndex].name
# get value by using: variables[myIndex].value
# variables are values stored in branches from TTree
MLCONFIG.variables = TBranches.getAllMembers()
MLCONFIG.categories = Category.getAllMembers()
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

# couldn't use dictionary due to problems with pickling enums
MLCONFIG.datasetsList = [
    (Category.GenuineTau, [path.join(MLCONFIG.datasetsBasePath, 'GenuineTau')]), 
    (Category.FakeTau, [path.join(MLCONFIG.datasetsBasePath, 'FakeTau')])
    ]
# endregion small test dataset

# # region large test dataset
# MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseCephPath, 'datasets')
# MLCONFIG.outputPath = path.join(MLCONFIG.baseWorkPath, 'TauAnalyzer/ML/output_largeDataset_{}'.format(datetimeStr))

# MLCONFIG.plotsOutputPath = path.join(MLCONFIG.outputPath, "plots")
# from pathlib import Path
# Path(MLCONFIG.outputPath).mkdir(parents=True, exist_ok=True)
# Path(MLCONFIG.plotsOutputPath).mkdir(parents=True, exist_ok=True)

# # couldn't use dictionary due to problems with pickling enums
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

MLCONFIG.datasetsInfoSummary = getDatasetsInfoSummary(MLCONFIG) # get file-/eventcounts of datasets (can take several minutes)
MLCONFIG.trainEventsPerCategoryPerBatch = 50
MLCONFIG.batchSize = int(MLCONFIG.trainEventsPerCategoryPerBatch * len(MLCONFIG.categories))
MLCONFIG.stepsPerEpoch = int(MLCONFIG.datasetsInfoSummary.totalEventCount['train'] / MLCONFIG.batchSize)
# might need to change if datasets become too large!
MLCONFIG.validEventsPerCategoryPerBatch = MLCONFIG.datasetsInfoSummary.categoryWithMaxEventCount['valid'] # this is a tuple ! (eventCount, CategoryName)
MLCONFIG.validationSteps = 1

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
else:
    MLCONFIG.saveToJsonfile(MLCONFIG.outputPath, cfgOutputFileName)
    print("Config file has been created.")
