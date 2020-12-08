from utils.json_util import JsonSerializable
from utils.pprint_util import PPrintable

class MLConfig(JsonSerializable, PPrintable):
    def __init__(self):
        self.baseWorkPath = ""
        self.baseCephPath = ""
        self.datasetsBasePath = ""
        self.datasetsList = []
        self.datasetsInfoSummary = None
        self.outputPath = ""
        self.plotsOutputPath = ""
        self.variables = []
        self.categories = []
        self.encodeLabels_OneHot = True
        self.generateHistograms = True
        self.mlparametersetPath = ""
        self.mlparams = None
        self.trainEventsPerCategoryPerBatch = 50
        self.batchSize = 1
        self.validEventsPerCategoryPerBatch = 2000000
        self.validationSteps = 1
        self.stepsPerEpoch = 1


