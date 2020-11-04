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
import glob
from os import path
from sklearn.utils import shuffle
from classes.mlconfig import MLConfig
from classes.enums.category import Category
from classes.enums.tbranches import TBranches
from utils.bashcolors import bcolors


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

def rootTTree2numpy(rootFilePath):
    f = uproot4.open(rootFilePath)
    ttree = f['tauEDAnalyzer']['Events'] # TTree name: 'Events'
    return ttree.arrays(library="np")

def buildDataset(cfg: MLConfig):
    if not isinstance(cfg, MLConfig):
        raise TypeError

    inputs_ = [] 
    labels_ = []
    branchNames = [branch.name for branch in cfg.variables]

    for category, datasets in cfg.datasetsList: # category is the key and rootFiles is the value
        for datasetPath in datasets:
            rootFiles = [f for f in glob.glob(datasetPath + "/*.root", recursive=False)]
            for rootFile in rootFiles:
                #print(rootFile)
                tree = rootTTree2numpy(rootFile)
                entryCount = tree[branchNames[0]].shape[0]
                # ml_variable.name is the branch name of the TTree. Each branch has one float variable e.g. Tau_pt
                inputs_.append(np.vstack([np.array(tree[branch], dtype=np.float32) for branch in branchNames]).T)
                labels_.append(np.full((entryCount,), category.value, dtype=np.float32))

    # Stack all inputs_ vertically
    inputs_ = np.vstack(inputs_)

    # Stack all labels_ horizontally
    labels_ = np.hstack(labels_)

    # Sort all inputs/labels to have a clear separation for plotting histograms
    indices = np.argsort(labels_)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    del indices

    # region Getting rid of Taus using python instead of C++ TauAnalyzer with dxy = -999 and decayModes 5,6
    # dxy_selectionMask = inputs_[:, int(TBranches.Tau_dxy)] > -990
    # decayMode5_selectionMask = inputs_[:, int(TBranches.Tau_decayMode)] != 5
    # decayMode6_selectionMask = inputs_[:, int(TBranches.Tau_decayMode)] != 6
    # selectionMask = np.logical_and(np.logical_and(dxy_selectionMask, decayMode5_selectionMask), decayMode6_selectionMask)
    # inputs_ = inputs_[selectionMask] 
    # labels_ = labels_[selectionMask] 
    # unique, counts = np.unique(selectionMask, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    # unique, counts = np.unique(inputs_[:,int(TBranches.Tau_decayMode)], return_counts=True)
    # print("Decay modes and their counts: ")
    # print(dict(zip(unique, counts)))

    print("Input shape: ", inputs_.shape)
    print("Labels shape: ", labels_.shape)
    # endregion

    categorySliceIndices = getCategorySliceIndicesFromSorted1DArray(labels_, cfg.categories)
    print("Categories that are being plotted with their corresponding slice indices: ")
    print(categorySliceIndices)

    if(cfg.encodeLabels_OneHot):
        labels_ = tf.keras.utils.to_categorical(labels_)

    # print(inputs_[:,:1].min()) # shows min Tau_pt
    # print(inputs_[:,:1].max()) # shows max Tau_pt

    # Generate Histograms
    # get slice indices of each category e.g. all indices of genuine taus stored as tuple (beginningIndex, EndIndex)
    if(cfg.generateHistograms):
        from pathlib import Path
        Path(cfg.plotsOutputPath).mkdir(parents=True, exist_ok=True)
        plotHistograms(inputs_, labels_, categorySliceIndices, cfg.plotsOutputPath, nBins=30)

    # Shuffle inputs_/labels_ for training
    indices = np.arange(labels_.shape[0])
    np.random.shuffle(indices)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    
    return inputs_, labels_

def plotHistograms(data, labels, categorySliceIndices, outputDirPath, nBins=99):
    # iterate through all tbranches
    for i in range(data.shape[1]):
        variable = TBranches(float(i))
        #iterate through all categories
        for j in range(len(categorySliceIndices)):
            category = categorySliceIndices[j][0]
            beginSliceIndex = categorySliceIndices[j][1]
            endSliceIndex = categorySliceIndices[j][2]
            histoData = data[beginSliceIndex:endSliceIndex, i]
            p = np.percentile(histoData, [1, 99])
            bins = np.linspace(p[0], p[1], nBins)
            plt.hist(histoData, bins=bins, alpha=0.7, label=category.name, histtype="step")

        plt.title(variable.name)
        plt.ylabel("frequency")
        logScaleVariables = (TBranches.Tau_ecalEnergy, TBranches.Tau_hcalEnergy, TBranches.Tau_mass)
        if(variable in logScaleVariables):
            plt.yscale("log")
        plt.legend(loc='upper right')
        from os import path
        filename = "Histo_{}.png".format(variable.name)
        outputFilePath = path.join(outputDirPath, filename)
        plt.savefig(outputFilePath)
        plt.clf()

def getCategorySliceIndicesFromSorted1DArray(sorted1DArray, categoryList):
    """
    Get and return list of tuples of the beginning and ending index of a category-slice in the sortedArray
    slice: (beginIndex,endIndex, elementCount) beginIndex is inclusive, endIndex is exlusive! 
    """
    categorySplitIndices = []
    for i in range(len(categoryList)):
        categorySplitIndices.append(np.searchsorted(sorted1DArray, categoryList[i].value, side='left'))
    categorySplitIndices.append(sorted1DArray.shape[0]) # append index of very last element + 1

    categorySlices = []
    for i in range(len(categorySplitIndices) - 1):
        categorySlices.append(
                (
                    Category(sorted1DArray[categorySplitIndices[i]]), 
                    categorySplitIndices[i], 
                    categorySplitIndices[i+1],
                    categorySplitIndices[i+1]-categorySplitIndices[i]
                )
            )

    return categorySlices
# endregion ######### Methods ######### 


print("\n" + bcolors.OKGREEN + bcolors.BOLD + "########## BEGIN PYTHON SCRIPT ############" + bcolors.ENDC)
# region ######### Get dataset from root files with uproot4 ######### 
cfg = MLConfig.loadFromJsonfile('cfgWithSmallDataset.json')

inputs, labels = buildDataset(cfg)
from sklearn.model_selection import train_test_split
inputs_train, inputs_testAndvalidation, labels_train, labels_testAndvalidation = train_test_split(inputs, labels, test_size=0.3, random_state=0)
del inputs, labels
inputs_validation, inputs_test, labels_validation, labels_test = train_test_split(inputs_testAndvalidation, labels_testAndvalidation, test_size=0.5, random_state=0)
# endregion ######### Get dataset from root files with uproot4 ######### 


# region ######### Tensorflow / Keras ######### 
# region ######### NN Model ######### 
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(len(cfg.variables),), name="inputLayer"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(len(cfg.variables),), name="dense_1"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(len(cfg.variables),), name="dense_2"))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="predictions"))
# endregion ######### NN Model ######### 

model.summary()
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# region ######### Training ######### 
history = model.fit(
    inputs_train, 
    labels_train, 
    batch_size=100,
    epochs=10,
    validation_data=(inputs_validation, labels_validation)
)
# endregion ######### Training ######### 



# NN output plot
predictions = model.predict(inputs_test)
#print(predictions)

if(cfg.encodeLabels_OneHot):
    # TODO: check which order is actually signal (genuineTau) and which are background (fakeTau)
    genuineTau_decisions = predictions[:,0]
    fakeTau_decisions = predictions[:,1]

    plt.hist(fakeTau_decisions, color='red', label='fake', 
            histtype='step', # lineplot that's unfilled
            density=True ) # normalize to form a probability density
    plt.hist(genuineTau_decisions, color='blue', label='genuine', 
            histtype='step', # lineplot that's unfilled
            density=True, # normalize to form a probability density
            linestyle='--' )
    plt.xlabel('Neural Network output') # add x-axis label
    plt.ylabel('Arbitrary units') # add y-axis label
    plt.legend() # add legend
    plt.savefig(path.join(cfg.plotsOutputPath, "NN_output.png"))
    plt.clf()

from sklearn.metrics import roc_curve, auc
# most tutorials slice the prediction for whatever reason with [:,1] but why?
# predictions_ = predictions[:, 1]

fpr, tpr, _ = roc_curve(labels_test.argmax(axis=1), predictions.argmax(axis=1))

roc_auc = auc(fpr, tpr) # area under curve (AUC), ROC = Receiver operating characteristic
plt.plot(fpr, tpr, label='ROC (area = %0.2f)'%(roc_auc)) # plot test ROC curve
plt.plot([0, 1], # x from 0 to 1
         [0, 1], # y from 0 to 1
         '--', # dashed line
         color='grey', label='Luck')

plt.xlabel('False Positive Rate') # x-axis label
plt.ylabel('True Positive Rate') # y-axis label
plt.title('Receiver operating characteristic (ROC) curve') # title
plt.legend() # add legend
plt.grid() # add grid
plt.savefig(path.join(cfg.plotsOutputPath, "ROC_Curve.png"))
plt.clf()

print("\n")
print(history.history)
# endregion ######### Tensorflow / Keras ######### 

print(bcolors.OKGREEN + bcolors.BOLD + "########## END PYTHON SCRIPT ############\n" + bcolors.ENDC)
