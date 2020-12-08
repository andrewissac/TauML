import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from os import path
from classes.enums.datamode import Datamode
from classes.enums.category import Category
from classes.enums.tbranches import TBranches
from classes.mlconfig import MLConfig
from classes.mlparameters import MLParameterset
from utils.rootfile_util import rootTTree2numpy

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--mlconfigfile", required=True, type=str)
parser.add_argument("-model", "--kerasmodel", required=True, type=str)
args = parser.parse_args()

def getDatasetSplit(cfg: MLConfig, datamode: Datamode):
    if not isinstance(cfg, MLConfig):
        raise TypeError

    inputs_ = [] 
    labels_ = []
    branchNames = [branch.name for branch in cfg.variables]

    for category, datasets in cfg.datasetsList:
        for datasetPath in datasets:
            print(datasetPath)
            print(path.join(datasetPath, datamode.name, "/*.root"))
            rootFiles = [f for f in glob.glob(path.join(datasetPath, datamode.name, "*.root"), recursive=False)]
            for rootFile in rootFiles:
                tree = rootTTree2numpy(rootFile)
                entryCount = tree[branchNames[0]].shape[0]
                inputs_.append(np.array([np.array(tree[branch], dtype=np.float32) for branch in branchNames]).T)
                labels_.append(np.full((entryCount,), category.value, dtype=np.float32))

    # Stack all inputs_ vertically
    inputs_ = np.vstack(inputs_)

    # Stack all labels_ horizontally
    labels_ = np.hstack(labels_)

    # Sort all inputs/labels to have a clear separation for plotting histograms
    # indices = np.argsort(labels_)
    # inputs_ = inputs_[indices]
    # labels_ = labels_[indices]
    # del indices

    labels_ = tf.keras.utils.to_categorical(labels_)

    # Shuffle inputs_/labels_ for training
    indices = np.arange(labels_.shape[0])
    np.random.shuffle(indices)
    inputs_ = inputs_[indices]
    labels_ = labels_[indices]
    
    return inputs_, labels_

cfg = MLConfig.loadFromJsonfile(args.mlconfigfile)
model = tf.keras.models.load_model(args.kerasmodel)
inputs_test, labels_test = getDatasetSplit(cfg, Datamode.test)

predictions = model.predict(inputs_test)

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

# print("\n")
# print(history.history)

# # Plot accuracy of NN
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(path.join(cfg.plotsOutputPath, "model_accuracy.png"))
# plt.clf()
# # Plot loss of NN
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(path.join(cfg.plotsOutputPath, "model_loss.png"))
# plt.clf()


# # evaluate the model
# _, train_acc = model.evaluate(inputs_train, labels_train, verbose=1)
_, test_acc = model.evaluate(inputs_test, labels_test, verbose=1)
print('Test: %.3f' % (test_acc))