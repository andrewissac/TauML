from classes.mlparameters import MLParameterset, EarlyStopping, ModelCheckpoint
from classes.enums.activation import Activation
from classes.enums.loss import LossFunction
from classes.enums.optimizer import Optimizer

Params = MLParameterset()

Params.learningrate = 0.000001
Params.inputlayer = { 'name': 'input', 'shape': (9,)}
Params.hiddenlayers = [
    {'name': 'dense01', 'type': 'dense', 'nodes': 64 ,'activation': Activation.relu},
    {'name': 'dense02', 'type': 'dense', 'nodes': 32 ,'activation': Activation.relu}
    ]
Params.outputlayer = { 'name': 'predictions', 'type': 'dense', 'nodes': 2, 'activation': Activation.softmax }
Params.batchsize = 32
Params.epochs = 100
Params.lossfunction = LossFunction.CategoricalCrossentropy
Params.lossfunction_fromLogits = True
Params.optimizer = Optimizer.Adam
Params.EarlyStopping = EarlyStopping(
    enabled=True, 
    monitor='val_accuracy', 
    patience=5, 
    verbose=1, 
    min_delta=0.0005)
Params.ModelCheckpoint = ModelCheckpoint(
    filepath='model.epoch-{epoch:02d}-val_loss-{val_loss:.4f}.h5', 
    enabled=True, 
    monitor='val_loss', 
    save_best_only=True, 
    verbose = True, 
    mode='auto'
    )

Params.saveToJsonfile('/work/aissac/TauAnalyzer/ML/parametersets', 'mlparams0001.json')