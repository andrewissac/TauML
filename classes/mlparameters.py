from utils.json_util import JsonSerializable
from utils.pprint_util import PPrintable

class MLParameterset(JsonSerializable):
    def __init__(self):
        # some random default values
        self.inputlayer = {}
        self.hiddenlayers = {} 
        self.outputlayer = {}
        self.batchsize = 16
        self.epochs = 10
        self.lossfunction = None
        self.optimizer = None
        self.earlystopping = None
        self.modelcheckpoint = None
        self._monitorStrings = ['train_loss', 'val_loss', 'train_mae', 'val_mae', 'train_accuracy', 'val_accuracy']
        self._modeStrings = ['auto', 'min', 'max']
    
    def buildSequentialKerasModel(self):
        import tensorflow as tf
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=self.inputlayer['shape'], name=self.inputlayer['name']))

        for hlayer in self.hiddenlayers:
            if hlayer['type'] == 'dense':
                model.add(tf.keras.layers.Dense(hlayer['nodes'], activation=hlayer['activation'], name=hlayer['name']))

        if self.outputlayer['type'] == 'dense':
            model.add(tf.keras.layers.Dense(self.outputlayer['nodes'], activation=self.outputlayer['activation'], name=self.outputlayer['name']))

        return model

    def __str__(self): # not inheriting from PPrintable, because tf functions would not print with all properties but with address
        return self.toJsonString()