class MLConfig():
    def __init__(self):
        self.baseWorkPath = ""
        self.baseCephPath = ""
        self.datasetsBasePath = ""
        self.datasetsDic = {}
        self.plotsOutputPath = ""
        self.variables = []
        self.categories = []
        self.encodeLabels_OneHot = True
        self.generateHistograms = True
