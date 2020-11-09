from utils.json_util import JsonSerializable
from utils.pprint_util import PPrintable
from .enums.optimizer import Optimizer
from .enums.activation import Activation
from .enums.loss import LossFunction

class MLParameterset(JsonSerializable, PPrintable):
    def __init__(self):
        # some random default values
        self.learningrate = 0.001
        self.inputlayer = {}
        self.hiddenlayers = {} 
        self.outputlayer = {}
        self.batchsize = 16
        self.epochs = 10
        self.lossfunction = LossFunction.CategoricalCrossentropy
        self.lossfunction_fromLogits = True
        self.optimizer = Optimizer.Adam
        self.EarlyStopping = EarlyStopping()
        self.ModelCheckpoint = ModelCheckpoint()
        self._monitorStrings = ['train_loss', 'val_loss', 'train_mae', 'val_mae', 'train_accuracy', 'val_accuracy']
        self._modeStrings = ['auto', 'min', 'max']

class EarlyStopping(PPrintable):
    def __init__(self, enabled = True, monitor = 'val_accuracy', patience = 5, verbose = 1, min_delta = 0.0001):
        self.enabled = enabled
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta

class ModelCheckpoint(PPrintable):
    def __init__(self, filepath='', enabled = True, monitor = 'val_loss', save_best_only = True, verbose = 1, mode = 'auto'):
        self.enabled = enabled
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.mode = mode