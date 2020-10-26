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

def rootTTree2numpy(path_, rootFile):
    f = uproot4.open(path.join(path_, rootFile))
    ttree = f['tauEDAnalyzer']['Events'] # TTree name: 'Events'
    return ttree.arrays(library="np")

def buildDataset(cfg: MLConfig):
    if not isinstance(cfg, MLConfig):
        raise TypeError

    inputs_ = [] 
    labels_ = []
    branchNames = [branch.name for branch in cfg.variables]

    for category, rootFilesList in cfg.rootFilesDictionary.items(): # category is the key and rootFiles is the value
        for rootFile in rootFilesList:
            tree = rootTTree2numpy(cfg.rootFilesDir, rootFile)
            entryCount = tree[branchNames[0]].shape[0]
            # ml_variable.name is the branch name of the TTree. Each branch has one float variable e.g. Tau_pt
            inputs_.append(np.vstack([np.array(tree[branch], dtype=np.float32) for branch in branchNames]).T)
            labels_.append(np.full((entryCount,), category.value, dtype=np.float32))

    # Stack all inputs_ vertically
    inputs_ = np.vstack(inputs_)

    # Stack all labels_ horizontally
    labels_ = np.hstack(labels_)

    # Sort all inputs/labels to have a clear separation
    indices = np.argsort(labels_)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]

    # region Getting rid of Taus with dxy = -999
    # selectionMask = inputs_[:, int(TBranches.Tau_dxy)] < -900
    # inputs_ = inputs_[selectionMask] # pruned for dxy < -900
    # labels_ = labels_[selectionMask] # pruned for dxy < -900

    # print("Input shape: ", inputs_.shape)
    # print("Labels shape: ", labels_.shape)
    # endregion

    # TODO: SORT LABELS BY ARGSORT THEN USE THE INDICES TO SORT INPUTS

    if(cfg.encodeLabels_OneHot):
        labels_ = tf.keras.utils.to_categorical(labels_)

    # print(inputs_[:,:1].min()) # shows min Tau_pt
    # print(inputs_[:,:1].max()) # shows max Tau_pt

    # Generate Histograms
    # get slice indices of each category e.g. all indices of genuine taus stored as tuple (beginningIndex, EndIndex)
    if(cfg.generateHistograms):
        categorySliceIndices = getCategorySliceIndicesFromSortedArray(labels_, cfg.encodeLabels_OneHot)
        print("Categories that are being plotted with their corresponding slice indices: ")
        print(categorySliceIndices)
        plotHistograms(inputs_, labels_, categorySliceIndices, cfg.plotsOutputDir, nBins=30)

    # Shuffle inputs_/labels_
    indices = np.arange(labels_.shape[0])
    np.random.shuffle(indices)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    
    return inputs_, labels_

def plotHistograms(data, labels, categorySliceIndices, outputDirPath, nBins=99):
    # TODO: X-Axis label
    for i in range(data.shape[1]):
        variable = TBranches(float(i))
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
    
def getCategorySplitIndicesFromSorted_OneHotArray(sortedOneHotArray):
    """
    Suppose myArr is a sorted (!) oneHotArray that is sorted:
    myArr = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    myArr.shape[1] # this will return the count of categories, in this example 2
    np.searchsorted(myArray[:,0], 1.0, side='left') # this will return the index of the first occurence of [1.0, 0.0], in this example: 0
    np.searchsorted(myArray[:,1], 1.0, side='left') # this will return the index of the first occurence of [0.0, 1.0], in this example: 3
    -> loop over the categories will return all indices where the categories change.
    """
    entryCount = sortedOneHotArray.shape[0]
    categoryCount = sortedOneHotArray.shape[1]

    categorySplitIndices = []
    for i in range(categoryCount):
        categorySplitIndices.append(np.searchsorted(sortedOneHotArray[:,i], 1.0, side='left'))
    categorySplitIndices.append(entryCount) # append index of very last element + 1

    return categorySplitIndices

def getCategorySplitIndicesFromSorted_1DArray(sorted1DArray):
    """
    Suppose myArr is a sorted (!) 1D Array :
    myArr = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    np.unique(..) will find the count of categories, in this example 4
    indices where the category changes are simply computed by adding their count
    """
    categories, counts = np.unique(sorted1DArray, return_counts=True)
    print(categories, counts)

    categorySplitIndices = []
    splitIndex = 0
    for i in range(len(categories)):
        categorySplitIndices.append(splitIndex)
        splitIndex += counts[i]
    categorySplitIndices.append(np.sum(counts)) # append index of very last element + 1
    
    return categorySplitIndices

def getCategorySliceIndicesFromSortedArray(sortedArray, IsOneHotArray):
    """
    Get and return list of tuples of the beginning and ending index of a category-slice in the sortedArray
    slice: (beginIndex,endIndex) beginIndex is inclusive, endIndex is exlusive! 
    """

    categorySplitIndices = []
    if(IsOneHotArray):
        categorySplitIndices = getCategorySplitIndicesFromSorted_OneHotArray(sortedArray)
    else:
        categorySplitIndices = getCategorySplitIndicesFromSorted_1DArray(sortedArray)

    categorySlices = []
    for i in range(len(categorySplitIndices) - 1):
        if(IsOneHotArray):
            categorySlices.append(
                    (
                        Category.oneHotVectorToEnum(sortedArray[categorySplitIndices[i]]), 
                        categorySplitIndices[i], 
                        categorySplitIndices[i+1]
                    )
                )
        else:
            categorySlices.append(
                    (
                        Category(sortedArray[categorySplitIndices[i]]), 
                        categorySplitIndices[i], 
                        categorySplitIndices[i+1]
                    )
                )

    return categorySlices
# endregion ######### Methods ######### 


print("\n" + bcolors.OKGREEN + bcolors.BOLD + "########## BEGIN PYTHON SCRIPT ############" + bcolors.ENDC)

# region ######### Get dataset from root files with uproot4 ######### 
# ml_variables are values stored in branches from TTree
from utils.mlconfig_utils import generateMLConfig
cfg = generateMLConfig()

#inputs, labels = buildDataset(cfg.rootFilesDir, cfg.rootFilesDictionary, cfg.variables, histogramOutputDirPath=cfg.plotsOutputDir, encodelabels_OneHot=cfg.encodeLabels_OneHot, generateHistograms=cfg.generateHistograms)
inputs, labels = buildDataset(cfg)
from sklearn.model_selection import train_test_split
inputs_train, inputs_testAndvalidation, labels_train, labels_testAndvalidation = train_test_split(inputs, labels, test_size=0.3, random_state=0)
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
    plt.savefig(path.join(cfg.plotsOutputDir, "NN_output.png"))
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
plt.savefig(path.join(cfg.plotsOutputDir, "ROC_Curve.png"))
plt.clf()

print("\n")
print(history.history)
# endregion ######### Tensorflow / Keras ######### 

print(bcolors.OKGREEN + bcolors.BOLD + "########## END PYTHON SCRIPT ############\n" + bcolors.ENDC)
