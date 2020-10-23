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
    # Since all labels are appended in the above for loop using category.value -> labels are always sorted!
    labels_ = np.hstack(labels_)
    #uniques, ids = np.unique(labels_, return_inverse=True)


    if(encodelabels_OneHot):
        labels_ = tf.keras.utils.to_categorical(labels_)

    # print(inputs_[:,:1].min()) # shows min Tau_pt
    # print(inputs_[:,:1].max()) # shows max Tau_pt

    # Generate Histograms
    # get slice indices of each category e.g. all indices of genuine taus stored as tuple (beginningIndex, EndIndex)
    categorySliceIndices = getCategorySliceIndicesFromSortedArray(labels_, encodelabels_OneHot)
    plotHistogram(inputs_, labels_, categorySliceIndices)

    # Shuffle inputs_/labels_
    indices = np.arange(labels_.shape[0])
    np.random.shuffle(indices)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    
    return inputs_, labels_

def plotHistogram(data, labels, categorySliceIndices):
    # TODO: BINS + AXIS 
    for i in range(data.shape[1]):
        for j in range(len(categorySliceIndices)):
            category = categorySliceIndices[j][0]
            beginSliceIndex = categorySliceIndices[j][1]
            endSliceIndex = categorySliceIndices[j][2]
            plt.hist(data[beginSliceIndex:endSliceIndex, i], bins=100, alpha=0.5, label=category.name)
        plt.legend(loc='upper right')
        plt.savefig("Histo_{}.png".format(cfg.TBranches(float(i)).name))
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
                        cfg.mapOneHotVectorToMLCategoryEnum(sortedArray[categorySplitIndices[i]]), 
                        categorySplitIndices[i], 
                        categorySplitIndices[i+1]
                    )
                )
        else:
            categorySlices.append(
                    (
                        cfg.TBranches(sortedArray[categorySplitIndices[i]]), 
                        categorySplitIndices[i], 
                        categorySplitIndices[i+1]
                    )
                )

    return categorySlices
# endregion ######### Methods ######### 


print("\n########## BEGIN PYTHON SCRIPT ############")

# region ######### Get dataset from root files with uproot4 ######### 
# ml_variables are values stored in branches from TTree
inputs, labels = buildDataset(cfg.testData_basepath, cfg.fileDic, cfg.ml_variables)
from sklearn.model_selection import train_test_split
inputs_train, inputs_validation, labels_train, labels_validation = train_test_split(inputs, labels, test_size=0.1, random_state=0)
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
# history = model.fit(
#     inputs, 
#     labels, 
#     batch_size=100,
#     epochs=10,
#     validation_data=(inputs_validation, labels_validation)
# )
# endregion ######### Training ######### 

print("\n")
print(history.history)
# endregion ######### Tensorflow / Keras ######### 

print("########## END PYTHON SCRIPT ############\n")
