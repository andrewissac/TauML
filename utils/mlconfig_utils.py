import classes.mlconfig as cfg
from classes.enums.category import Category
from classes.enums.tbranches import TBranches


def generateMLConfig():
    MLCONFIG = cfg.MLConfig()
    # region ######### Paths and files ######### 
    MLCONFIG.baseWorkPath = '/work/aissac/'
    MLCONFIG.baseCephPath = '/ceph/aissac/'
    from os import path
    #MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseWorkPath, 'TestRootFiles')
    MLCONFIG.datasetsBasePath = path.join(MLCONFIG.baseCephPath, 'mergedROOTFiles')
    #MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_dxy-999_decayMode5_6_PrunedInAnalyzer")
    #MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_dxy-999_decayMode5_6_PrunedInPython")
    #MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots")
    MLCONFIG.plotsOutputPath = path.join(MLCONFIG.baseWorkPath, "TauAnalyzer/ML/output_plots_all")
    from pathlib import Path
    Path(MLCONFIG.plotsOutputPath).mkdir(parents=True, exist_ok=True)
    MLCONFIG.datasetsDic = { 
        # Category.GenuineTau : [path.join(MLCONFIG.datasetsBasePath, 'GenuineTau')], 
        # Category.FakeTau : [path.join(MLCONFIG.datasetsBasePath, 'FakeTau')] 
        Category.GenuineTau : [
            path.join(MLCONFIG.datasetsBasePath, 'output__DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1_MINIAODSIM')
            ], 
        Category.FakeTau : [
            path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14_ext1-v2_MINIAODSIM'),
            path.join(MLCONFIG.datasetsBasePath, 'output__WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3_MINIAODSIM')
            ] 
        }
    # endregion ######### Paths and files ######### 

    # get names (e.g. for a dict) by using: variables[myIndex].name
    # get value by using: variables[myIndex].value
    # variables are values stored in branches from TTree
    MLCONFIG.variables = [member for name, member in TBranches.getAllMembers()]
    MLCONFIG.categories = [member for name, member in Category.getAllMembers()]

    return MLCONFIG
    # TODO: SAVE ML CONFIG TO JSON AND READ CONFIG FROM JSON!