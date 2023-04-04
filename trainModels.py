import os

import numpy
from tensorflow import keras

from labels import naturalClassesFromCoarse
from model import trainModel

# Hide tensorflow's info level logs, warnings and errors will still show
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

cifar = keras.datasets.cifar100
(trainingDataFine, trainingLabelsFine), (testDataFine, testLabelsFine) = cifar.load_data(label_mode='fine')
(trainingDataCoarse, trainingLabelsCoarse), (testDataCoarse, testLabelsCoarse) = cifar.load_data(label_mode='coarse')

trainingLabelsNatural = []

for i in range(len(trainingLabelsCoarse)):
    if naturalClassesFromCoarse.count(trainingLabelsCoarse[i]):
        trainingLabelsNatural.append([0])
    else:
        trainingLabelsNatural.append([1])  
        
trainingLabelsNatural = numpy.array(trainingLabelsNatural)  
        
trainModel('class', trainingDataCoarse, trainingLabelsCoarse)
trainModel('subclass', trainingDataFine, trainingLabelsFine)
trainModel('superclass', trainingDataCoarse, trainingLabelsNatural)