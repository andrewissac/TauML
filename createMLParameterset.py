from classes.mlparameters import MLParameterset
from os import path
from pathlib import Path
import tensorflow as tf

Params = MLParameterset()

# model parameter
Params.inputlayer = { 'name': 'input', 'shape': (9,)}
Params.hiddenlayers = [
    {'name': 'dense01', 'type': 'dense', 'nodes': 32 ,'activation': tf.nn.relu},
    {'name': 'dense02', 'type': 'dense', 'nodes': 16 ,'activation': tf.nn.relu}
    ]
Params.outputlayer = { 'name': 'predictions', 'type': 'dense', 'nodes': 2, 'activation': tf.nn.softmax }

# training parameter
Params.eventsPerClassPerBatch = 50
Params.lossfunction = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
Params.optimizer = tf.keras.optimizers.Adam(lr=0.0001)
Params.earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.0005)
Params.modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model', monitor='val_loss', save_best_only=True, verbose=1)
Params.csvlogger = tf.keras.callbacks.CSVLogger(filename='results', separator=',', append=False)

paramsOutputFileName = 'mlparams0001.json'
paramsOutputPath = '/work/aissac/TauAnalyzer/ML/parametersets'
paramsFile = Path(path.join(paramsOutputPath, paramsOutputFileName))

if paramsFile.is_file():
    userInput = input('MLParameterset file already exists! Do you want to overwrite the config file? (y/n)\n')
    userInput = userInput.lower()
    if userInput == 'y':
        Params.saveToJsonfile(paramsOutputPath, paramsOutputFileName)
        print("MLParameterset file overwritten.")
    else:
        print('MLParameterset file was not overwritten.')
        pass
else:
    Params.saveToJsonfile(paramsOutputPath, paramsOutputFileName)
    print("MLParameterset file has been created.")
