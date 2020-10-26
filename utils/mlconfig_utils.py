import classes.mlconfig as cfg
from classes.enums.category import Category
from classes.enums.tbranches import TBranches


def generateMLConfig():
    MLCONFIG = cfg.MLConfig()

    # region ######### Paths and files ######### 
    MLCONFIG.basePath = '/work/aissac/'
    from os import path
    MLCONFIG.rootFilesDir = path.join(MLCONFIG.basePath, 'TestRootFiles')
    MLCONFIG.plotsOutputDir = path.join(MLCONFIG.basePath, "TauAnalyzer/ML/output_plots_test")
    from pathlib import Path
    Path(MLCONFIG.plotsOutputDir).mkdir(parents=True, exist_ok=True)
    MLCONFIG.rootFilesDictionary = { 
        Category.GenuineTau : ['testDYJetsToLL.root'], 
        Category.FakeTau : ['testWJetsToLNu.root'] 
        }
    # endregion ######### Paths and files ######### 

    # get names (e.g. for a dict) by using: variables[myIndex].name
    # get value by using: variables[myIndex].value
    # variables are values stored in branches from TTree
    MLCONFIG.variables = [member for name, member in TBranches.getAllMembers()]
    MLCONFIG.categories = [member for name, member in Category.getAllMembers()]

    return MLCONFIG
    # TODO: SAVE ML CONFIG TO JSON AND READ CONFIG FROM JSON!