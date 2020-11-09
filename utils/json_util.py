import jsonpickle

class JsonSerializable():
    def toJsonString(self):
        jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
        return jsonpickle.encode(self)

    def saveToJsonfile(self, outputPath, outputFilename):
        jsonString = self.toJsonString()
        from os import path
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