from datetime import timedelta
import os
import random
import time
import numpy

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from buildModel import buildModel
from helpers import reportDevice
from hyperparameters import hyperparameters
from stringLabels import cifar100FineLabels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Hide tensorflow's info level logs, warnings and errors will still show

reportDevice()

cifar = keras.datasets.cifar100
(trainingData, trainingLabels), (testData, testLabels) = cifar.load_data(label_mode='fine')

model = buildModel(hyperparameters)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learningRate']), 
    # Pull into hyperparams
    loss=hyperparameters['lossFunction'], 
    metrics=hyperparameters['trainingMetrics']
)

imageIndex = random.randint(0, len(testData))
image = testData[imageIndex]
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

prediction = model(image)
print('\nPredicted label:', cifar100FineLabels[numpy.argmax(prediction)], 'with', round((numpy.amax(prediction) * 100), 2), 'percent confidence')
print('Actual label:', cifar100FineLabels[testLabels[imageIndex][0]])

earlyStopping = EarlyStopping(monitor='val_loss', patience=3) # Ends early if validation loss stops decreasing for 3 epochs
start = time.perf_counter()
model.fit(
    x=trainingData, 
    y=trainingLabels, 
    epochs=hyperparameters['epochs'],
    validation_split=hyperparameters['validationSplit'],
    shuffle=True,
    callbacks=[earlyStopping]
)
secondsTaken = time.perf_counter() - start
print('\nTraining elapsed time:', timedelta(seconds=secondsTaken))
print('Time per epoch:', timedelta(seconds=secondsTaken / hyperparameters['epochs']))

# Prepare image could be pulled out?
imageIndex = random.randint(0, len(testData))
image = testData[imageIndex]
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

prediction = model(image)
print('\nPredicted label:', cifar100FineLabels[numpy.argmax(prediction)], 'with', round((numpy.amax(prediction) * 100), 1), 'percent confidence')
print('Actual label:', cifar100FineLabels[testLabels[imageIndex][0]])


# Wrap the training and evalution/prediction into functions, design it to be imported and run from 
# interactive python

# Wrap prediction into an agent that loads all three models with my trained weights and runs them with an input

# The buildModel and TrainModel functions might be identical other than loading in different labels,
# each should have it's own hyperparams? label selector could be in hyperparams

# When code exists to save the weights for reuse, how can the store path be passed as a param? in hyperparams?