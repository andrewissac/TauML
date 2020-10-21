import uproot4
import sys
import argparse
import copy
#import ROOT
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pickle
import MLConfig as cfg
from os import path
from sklearn.utils import shuffle

# region ######### Methods ######### 
def shuffleNumpyArrays(*args):
    """ 
    args must be numpy arrays + have the same length! 
    but somehow this does not change the arrays outside the function although numpy arrays are mutable (??)
    """
    try:
        indices_ = np.arange(args[0].shape[0])
        np.random.shuffle(indices_)
        for arr in args:
            arr = arr[indices_]
    except Exception as ex:
        print("shuffleNumpyArrays() - ", ex)

def rootTTree2numpy(path_, rootFile):
    f = uproot4.open(path.join(path_, rootFile))
    ttree = f['tauEDAnalyzer']['Events'] # TTree name: 'Events'
    return ttree.arrays(library="np")

def buildDataset(path_, fileDic, branchesToGetFromRootFiles, encodelabels_OneHot = True):
    inputs_ = [] 
    labels_ = []
    branchNames = [branch.name for branch in branchesToGetFromRootFiles]
    for category, rootFilesList in fileDic.items(): # category is the key and rootFiles is the value
        for rootFile in rootFilesList:
            tree = rootTTree2numpy(path_, rootFile)
            entryCount = tree[branchNames[0]].shape[0]
            # ml_variable.name is the branch name of the TTree. Each branch has one float variable e.g. Tau_pt
            inputs_.append(np.vstack([np.array(tree[branch], dtype=np.float32) for branch in branchNames]).T)
            labels_.append(np.full((entryCount,), category.value, dtype=np.float32))

    # Stack all inputs_ vertically
    inputs_ = np.vstack(inputs_)

    # Stack all labels_ horizontally
    labels_ = np.hstack(labels_)

    if(encodelabels_OneHot):
        labels_ = tf.keras.utils.to_categorical(labels_)

    # Shuffle inputs_/labels_
    indices = np.arange(labels_.shape[0])
    np.random.shuffle(indices)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    
    return inputs_, labels_
# endregion ######### Methods ######### 


print("\n########## BEGIN PYTHON SCRIPT ############")

# region ######### Get dataset from root files with uproot4 ######### 
# ml_variables are values stored in branches from TTree
inputs, labels = buildDataset(cfg.testData_basepath, cfg.fileDic, cfg.ml_variables)
validationEntryCount = -10000
inputs_validation = inputs[validationEntryCount:]
labels_validation = labels[validationEntryCount:]
inputs_train = inputs[:validationEntryCount]
label_train = inputs[:validationEntryCount]
# endregion ######### Get dataset from root files with uproot4 ######### 


# region ######### Tensorflow / Keras ######### 
# region ######### NN Model ######### 
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(len(cfg.ml_variables),), name="inputLayer"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(len(cfg.ml_variables),), name="dense_1"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(len(cfg.ml_variables),), name="dense_2"))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="predictions"))
# endregion ######### NN Model ######### 

model.summary()
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# region ######### Training ######### 
history = model.fit(
    inputs, 
    labels, 
    batch_size=100,
    epochs=10,
    validation_data=(inputs_validation, labels_validation)
)
# endregion ######### Training ######### 

print("\n")
print(history.history)
# endregion ######### Tensorflow / Keras ######### 

print("########## END PYTHON SCRIPT ############\n")
