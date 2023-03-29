import os
import random
import numpy

from tensorflow import keras

from model import trainModel
from helpers import reportDevice
from hyperparameters import superclassHyperparameters as hyperparameters
from labels import cifar100NaturalLabels, naturalClassesFromCoarse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Hide tensorflow's info level logs, warnings and errors will still show

reportDevice()

cifar = keras.datasets.cifar100
(trainingData, trainingLabels), (testData, testLabels) = cifar.load_data(label_mode='coarse')

for i in range(len(trainingLabels)):
    if naturalClassesFromCoarse.count(trainingLabels[i]):
        trainingLabels[i] = [0]
    else:
        trainingLabels[i] = [1]

for i in range(len(testLabels)):
    if naturalClassesFromCoarse.count(testLabels[i]):
        testLabels[i] = [0]
    else:
        testLabels[i] = [1]

model = trainModel('superclass', trainingData, trainingLabels, hyperparameters)

test_accuracy = model.evaluate(x=testData, y=testLabels)[1]
print('\nTest accuracy: %.5f' % test_accuracy)

# Prepare image could be pulled out?
imageIndex = random.randint(0, len(testData))
image = testData[imageIndex]
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

prediction = model(image)
print('\nPredicted label:', cifar100NaturalLabels[numpy.argmax(prediction)], 'with', round((numpy.amax(prediction) * 100), 1), 'percent confidence')
print('Actual label:   ', cifar100NaturalLabels[testLabels[imageIndex][0]])


# Wrap the training and evalution/prediction into functions, design it to be imported and run from 
# interactive python

# Wrap prediction into an agent that loads all three models with my trained weights and runs them with an input

# The buildModel and TrainModel functions might be identical other than loading in different labels,
# each should have it's own hyperparams? label selector could be in hyperparams

# When code exists to save the weights for reuse, how can the store path be passed as a param? in hyperparams?