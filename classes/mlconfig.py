import pprint
import jsonpickle
from os import path

class MLConfig():
    def __init__(self):
        self.baseWorkPath = ""
        self.baseCephPath = ""
        self.datasetsBasePath = ""
        self.datasetsList = []
        self.plotsOutputPath = ""
        self.variables = []
        self.categories = []
        self.encodeLabels_OneHot = True
        self.generateHistograms = True
    
    def __str__(self):
        return pprint.pformat(self.__dict__, indent=4)
    
    def toJsonString(self):
        jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
        return jsonpickle.encode(self)

    def saveToJsonfile(self, outputPath, outputFilename):
        jsonString = self.toJsonString()
        with open(path.join(outputPath, outputFilename), 'w') as jsonFile:
            jsonFile.write(jsonString)

    @classmethod
    def loadFromJsonString(cls, jsonString):
        return jsonpickle.decode(jsonString)

    @classmethod
    def loadFromJsonfile(cls, filePath):
        with open(filePath, 'r') as jsonFile:
            jsonString = jsonFile.read()
            return cls.loadFromJsonString(jsonString)


