from .enums.datamode import Datamode
from utils.json_util import JsonSerializable

class DatasetInfo(JsonSerializable):
    def __init__(self, category, datasetPath, rootfileInfos):
        self.category = category
        self.datasetPath = datasetPath
        self.rootfileInfos = rootfileInfos
        
        datamodes = Datamode.getAllMembers()
        self.fileCount = {'all' : sum(1 for x in rootfileInfos)}
        for datamode in datamodes:
            self.fileCount[datamode.name] = sum(1  for x in rootfileInfos if x.datamode == datamode)

        self.erroneousFileCount = {'all' : sum(1 for x in rootfileInfos if x.hadErrorOnOpening)}
        for datamode in datamodes:
            self.erroneousFileCount[datamode.name] = sum(1 for x in rootfileInfos if x.hadErrorOnOpening and x.datamode == datamode)

        self.eventCount = {'all' : sum(x.eventCount for x in rootfileInfos if not x.hadErrorOnOpening)}
        for datamode in datamodes:
            self.eventCount[datamode.name] = sum(x.eventCount for x in rootfileInfos if not x.hadErrorOnOpening and x.datamode == datamode)
        # rather a hack to not include thousands of lines into json file
        # comment line below out if specific rootfile info is needed
        self.rootfileInfos = [] 

class DatasetInfoSummary(JsonSerializable):
    def __init__(self, datasetInfoList):
        self.datasetInfoList = datasetInfoList
        datamodes = Datamode.getAllMembers()
        self.totalFileCount = {'all' : sum(x.fileCount['all'] for x in datasetInfoList)}
        for datamode in datamodes:
            self.totalFileCount[datamode.name] = sum(x.fileCount[datamode.name] for x in datasetInfoList)

        self.totalErroneousFileCount = {'all' : sum(x.erroneousFileCount['all'] for x in datasetInfoList)}
        for datamode in datamodes:
            self.totalErroneousFileCount[datamode.name] = sum(x.erroneousFileCount[datamode.name] for x in datasetInfoList)

        self.totalEventCount = {'all' : sum(x.eventCount['all'] for x in datasetInfoList)}
        for datamode in datamodes:
            self.totalEventCount[datamode.name] = sum(x.eventCount[datamode.name] for x in datasetInfoList)

        self.maxEventCount = {}
        for datamode in datamodes: # get event count from dataset with most events per datamode
            self.maxEventCount[datamode.name] = (max(x.eventCount[datamode.name] for x in datasetInfoList), self.getDatasetWithMaxEventCount(datamode.name).datasetPath)

        self.minEventCount = {}
        for datamode in datamodes: # get event count from dataset with least events per datamode
            self.minEventCount[datamode.name] = (min(x.eventCount[datamode.name] for x in datasetInfoList), self.getDatasetWithMinEventCount(datamode.name).datasetPath)

    def getDatasetWithMaxEventCount(self, mode):
        '''
        very naive "search", this approach assumes eventCounts per Dataset are unique 
        (very little chance that they are not unique OR purposely built to be of exact same size)
        '''
        maxEventCount = max(x.eventCount[mode] for x in self.datasetInfoList)
        for datasetInfo in self.datasetInfoList:
            if(maxEventCount == datasetInfo.eventCount[mode]):
                return datasetInfo

    def getDatasetWithMinEventCount(self, mode):
        '''
        very naive "search", this approach assumes eventCounts per Dataset are unique 
        (very little chance that they are not unique OR purposely built to be of exact same size)
        '''
        minEventCount = min(x.eventCount[mode] for x in self.datasetInfoList)
        for datasetInfo in self.datasetInfoList:
            if(minEventCount == datasetInfo.eventCount[mode]):
                return datasetInfo


class ROOTFileInfo(JsonSerializable):
    def __init__(self, filepath, datamode, hadErrorOnOpening, eventCount):
        self.filepath = filepath
        self.datamode = datamode
        self.hadErrorOnOpening = hadErrorOnOpening
        self.eventCount = eventCount