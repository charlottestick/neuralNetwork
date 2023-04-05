import os
import shutil
import time
from datetime import timedelta
from typing import Literal

from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import TensorBoard

from helpers import reportDevice
from hyperparameters import hyperparameters


def buildModel(numberOfClasses: int, hyperparameters: dict[str, any]):
    inputLayer = keras.Input(shape=(32, 32, 3)) # We need to create an input layer that matches our image size
    # The pretrained models are setup for larger image inputs, so we need to preprocess to resize
    resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
    dataAugmentationLayer = keras.Sequential(
        [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.2)]
    )(resizeLayer)


    baseModel = InceptionV3(include_top=False, weights=hyperparameters['transferLearningWeights'])
    print('Number of freezable InceptionV3 layers:', len(baseModel.layers))
    if hyperparameters['transferLearningWeights'] != None:
        for layer in baseModel.layers[:hyperparameters['frozenInceptionLayers']]:
            layer.trainable = False # Freeze lower original weights to prevent overfitting
    pretrainedLayer = baseModel(dataAugmentationLayer)
    
    untrainedLayer = keras.layers.GlobalAveragePooling2D()(pretrainedLayer)
    # untrainedLayer = keras.layers.Dense(2000, activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(1750, activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(1500, activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(1250, activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(1000, activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(750,  activation='relu')(untrainedLayer)
    untrainedLayer = keras.layers.Dense(500,  activation='relu')(untrainedLayer)
    untrainedLayer = keras.layers.Dense(250,  activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(200,  activation='relu')(untrainedLayer)
    untrainedLayer = keras.layers.Dense(150,  activation='relu')(untrainedLayer)
    # untrainedLayer = keras.layers.Dense(100,  activation='relu')(untrainedLayer) 
        
    dropoutLayer = keras.layers.Dropout(0.2)(untrainedLayer)
    
    outputLayer = keras.layers.Dense(numberOfClasses, activation='softmax')(dropoutLayer)
 
    return keras.Model(inputLayer, outputLayer)

def trainModel(classifier: Literal['superclass', 'class', 'subclass'], trainingData: list[list[int]], trainingLabels: list[list[int]]):
    numberOfClasses: Literal[2, 20, 100]
    
    if classifier == 'superclass':
        numberOfClasses = 2
    elif classifier == 'class':
        numberOfClasses = 20
    elif classifier == 'subclass':
        numberOfClasses = 100
    else:
        print('Invalid class type')
        return
    
    # Remove old logs first
    try:
        shutil.rmtree(os.getcwd() + '/logs/' + classifier + '/train', ignore_errors=True)
        shutil.rmtree(os.getcwd() + '/logs/' + classifier + '/validation', ignore_errors=True)
    except:
        print("Couldn't delete previous logs")
        
    
    model = buildModel(numberOfClasses, hyperparameters)
    model.summary()
    print('\n', classifier)
    print('\n', hyperparameters)
    reportDevice()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learningRate']), 
        loss=hyperparameters['lossFunction'], 
        metrics=hyperparameters['trainingMetrics']
    )
  
    tensorBoard = TensorBoard(log_dir=('./logs/' + classifier))
    start = time.perf_counter()
    model.fit(
        x=trainingData, 
        y=trainingLabels, 
        epochs=hyperparameters['epochs'],
        validation_split=hyperparameters['validationSplit'],
        shuffle=True,
        callbacks=[tensorBoard]
    )
    secondsTaken = time.perf_counter() - start
    print('\nTraining elapsed time:', timedelta(seconds=secondsTaken))
    print('Time per epoch:', timedelta(seconds=secondsTaken / hyperparameters['epochs']))

    savePath = './models/' + classifier
    model.save(savePath)