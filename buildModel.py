from typing import Dict
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

def buildModel(hyperparameters: Dict):
    inputLayer = keras.Input(shape=(32, 32, 3)) # We need to create an input layer that matches our image size
    # The pretrained models are setup for larger image inputs, so we need to preprocess to resize
    resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
    dataAugmentationLayer = keras.Sequential(
        [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(.2)]
    )(resizeLayer)


    baseModel = InceptionV3(include_top=False, weights=hyperparameters['transferLearningWeights'])
    # Can we only use the image net weights for frozen layers, override the rest with random?
    if hyperparameters['transferLearningWeights'] != None:
        for layer in baseModel.layers[:-hyperparameters['trainableInceptionLayers']]:
            layer.trainable = False # Freeze lower original weights to prevent overfitting
    pretrainedLayer = keras.layers.GlobalAveragePooling2D()(baseModel(dataAugmentationLayer))


    dropoutLayer = keras.layers.Dropout(.2)(pretrainedLayer)
    outputLayer = keras.layers.Dense(hyperparameters['numberOfClasses'], activation='softmax')(dropoutLayer)
 
    return keras.Model(inputLayer, outputLayer)