from utils.json_util import JsonSerializable
from utils.pprint_util import PPrintable

class MLConfig(JsonSerializable, PPrintable):
    def __init__(self):
        self.baseWorkPath = ""
        self.baseCephPath = ""
        self.datasetsBasePath = ""
        self.datasetsList = []
        self.outputPath = ""
        self.plotsOutputPath = ""
        self.variables = []
        self.categories = []
        self.encodeLabels_OneHot = True
        self.generateHistograms = True
        self.mlparametersetPath = ""
        self.mlparams = None


