from .enums.datamode import Datamode
from utils.json_util import JsonSerializable

class DatasetInfo(JsonSerializable):
    def __init__(self, category, datasetPath, rootfileInfos):
        self.category = category
        self.datasetPath = datasetPath
        self.rootfileInfos = rootfileInfos
        self.datamodes = Datamode.getAllMembers()
        self.fileCount = {'all' : sum(1 for x in rootfileInfos)}
        self.erroneousFileCount = {'all' : sum(1 for x in rootfileInfos if x.hadErrorOnOpening)}
        self.eventCount = {'all' : sum(x.eventCount for x in rootfileInfos if not x.hadErrorOnOpening)}
        for datamode in self.datamodes:
            self.fileCount[datamode.name] = sum(1  for x in rootfileInfos if x.datamode == datamode)
            self.erroneousFileCount[datamode.name] = sum(1 for x in rootfileInfos if x.hadErrorOnOpening and x.datamode == datamode)
            self.eventCount[datamode.name] = sum(x.eventCount for x in rootfileInfos if not x.hadErrorOnOpening and x.datamode == datamode)

        # rather a hack to not include thousands of lines into json file
        # comment line below out if specific rootfile info is needed
        self.rootfileInfos = [] 

class DatasetInfoSummary(JsonSerializable):
    def __init__(self, datasetInfoList):
        self.datasetInfoList = datasetInfoList
        self.datamodes = Datamode.getAllMembers()
        self.totalFileCount = {'all' : sum(x.fileCount['all'] for x in datasetInfoList)}
        self.totalErroneousFileCount = {'all' : sum(x.erroneousFileCount['all'] for x in datasetInfoList)}
        self.totalEventCount = {'all' : sum(x.eventCount['all'] for x in datasetInfoList)}
        self.categoryEventCount = {}
        self.categoryWithMaxEventCount = {}
        self.categoryWithMinEventCount = {}
        self.datasetWithMaxEventCount = {}
        self.datasetWithMinEventCount = {}
        for datamode in self.datamodes:
            self.totalFileCount[datamode.name] = sum(x.fileCount[datamode.name] for x in datasetInfoList)
            self.totalErroneousFileCount[datamode.name] = sum(x.erroneousFileCount[datamode.name] for x in datasetInfoList)
            self.totalEventCount[datamode.name] = sum(x.eventCount[datamode.name] for x in datasetInfoList)
            self.categoryEventCount[datamode.name] = self.getCategoryEventCount(datamode.name)
            self.categoryWithMaxEventCount[datamode.name] = self.getCategoryWithMinOrMaxEventCount(max, datamode.name)
            self.categoryWithMinEventCount[datamode.name] = self.getCategoryWithMinOrMaxEventCount(min, datamode.name)
            self.datasetWithMaxEventCount[datamode.name] = self.getDatasetWithMinOrMaxEventCount(max, datamode.name)
            self.datasetWithMinEventCount[datamode.name] = self.getDatasetWithMinOrMaxEventCount(min, datamode.name)

    def getCategoryEventCount(self, modeName: str):
        categoryEventCount = {}
        for datasetinfo in self.datasetInfoList:
            if not datasetinfo.category.name in categoryEventCount.keys():
                categoryEventCount[datasetinfo.category.name] = datasetinfo.eventCount[modeName]
            else:
                categoryEventCount[datasetinfo.category.name] += datasetinfo.eventCount[modeName]
        return categoryEventCount

    def getCategoryWithMinOrMaxEventCount(self, minOrMaxFunc, modeName: str):
        categoryEventCount = {}
        for datasetinfo in self.datasetInfoList:
            if not datasetinfo.category.name in categoryEventCount.keys():
                categoryEventCount[datasetinfo.category.name] = datasetinfo.eventCount[modeName]
            else:
                categoryEventCount[datasetinfo.category.name] += datasetinfo.eventCount[modeName]
        minOrMaxVal = minOrMaxFunc(categoryEventCount.values())
        minOrMaxKey = next((k for k,v in categoryEventCount.items() if v == minOrMaxVal), None)
        return (minOrMaxVal, minOrMaxKey)

    def getDatasetWithMinOrMaxEventCount(self, minOrMaxFunc, modeName: str):
        '''
        very naive "search", this approach assumes eventCounts per Dataset are unique 
        (very little chance that they are not unique OR purposely built to be of exact same size)
        '''
        minOrMaxEventCount = minOrMaxFunc(x.eventCount[modeName] for x in self.datasetInfoList)
        return (minOrMaxEventCount, next((x.datasetPath for x in self.datasetInfoList if x.eventCount[modeName] == minOrMaxEventCount), None))


class ROOTFileInfo(JsonSerializable):
    def __init__(self, filepath, datamode, hadErrorOnOpening, eventCount):
        self.filepath = filepath
        self.datamode = datamode
        self.hadErrorOnOpening = hadErrorOnOpening
        self.eventCount = eventCount