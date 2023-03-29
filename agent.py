import random

import numpy
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from labels import *

class Agent:
    def __init__(self) -> None:
        try:
            self.superclassModel = load_model('superclass')
            self.classModel = load_model('class')
            self.subclassModel = load_model('subclass')
        except:
            print("One or more model files weren't found, using InceptionV3 model")
            inputLayer = keras.Input(shape=(32, 32, 3))
            resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
            baseModel = InceptionV3()(resizeLayer)
            superclassOutputLayer = keras.layers.Dense(2, activation='softmax')(baseModel)
            classOutputLayer = keras.layers.Dense(20, activation='softmax')(baseModel)
            subclassOutputLayer = keras.layers.Dense(100, activation='softmax')(baseModel)
            self.superclassModel = keras.Model(inputLayer, superclassOutputLayer)
            self.classModel = keras.Model(inputLayer, classOutputLayer)
            self.subclassModel = keras.Model(inputLayer, subclassOutputLayer)

        cifar = keras.datasets.cifar100
        (_trainingData, _trainingLabels), (self.testData, self.testLabelsFine) = cifar.load_data(label_mode='fine')
        (_trainingData, _trainingLabels), (_testData, self.testLabelsCoarse) = cifar.load_data(label_mode='coarse')
        self.testLabelsNatural = []
        for i in range(len(self.testLabelsCoarse)):
            if naturalClassesFromCoarse.count(self.testLabelsCoarse[i]):
                self.testLabelsNatural.append([0])
            else:
                self.testLabelsNatural.append([1])


    def getRandomTestImages(self, numberOfImages: int):
        imageIndexes: list[int] = []
        for i in range(numberOfImages):
            imageIndex = random.randint(0, len(self.testData))
            imageIndexes.append(imageIndex)
        return imageIndexes


    def predict(self, imageIndexes: list[int]):
        for imageIndex in imageIndexes:
            image = self.testData[imageIndex]
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

            superclassPrediction = self.superclassModel(image)
            print('\nPredicted superclass:', cifar100NaturalLabels[numpy.argmax(superclassPrediction)], 'with', round((numpy.amax(superclassPrediction) * 100), 1), 'percent confidence')
            print('Actual superclass:   ', cifar100NaturalLabels[self.testLabelsNatural[imageIndex][0]])

            classPrediction = self.classModel(image)
            print('Predicted class:', cifar100CoarseLabels[numpy.argmax(classPrediction)], 'with', round((numpy.amax(classPrediction) * 100), 1), 'percent confidence')
            print('Actual class:   ', cifar100CoarseLabels[self.testLabelsCoarse[imageIndex][0]])

            subclassPrediction = self.subclassModel(image)
            print('Predicted subclass:', cifar100FineLabels[numpy.argmax(subclassPrediction)], 'with', round((numpy.amax(subclassPrediction) * 100), 1), 'percent confidence')
            print('Actual subclass:   ', cifar100FineLabels[self.testLabelsFine[imageIndex][0]])

    def generateConfusionMatrix(self, classType):
        if classType == 'superclass':
            model = self.superclassModel
            labelValues = cifar100NaturalLabels
            testLabels = self.testLabelsNatural
        elif classType == 'class':
            model = self.classModel
            labelValues = cifar100CoarseLabels
            testLabels = self.testLabelsCoarse
        elif classType == 'subclass':
            model = self.subclassModel
            labelValues = cifar100FineLabels
            testLabels = self.testLabelsFine
        else:
            print('Invalid class type')
            return
        
        targetClass = random.randrange(0, len(labelValues))

        confusionMatrix = {'targetClass': labelValues[targetClass], 'truePositive': 0, 'falsePositive': 0, 'trueNegative': 0, 'falseNegative': 0}
        print('Target class:', confusionMatrix['targetClass'])

        index = 0
        totalImages = len(self.testData)
        for image in self.testData:
            print('\r', index, 'out of', totalImages, end='')

            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            prediction = model(image)
            predictedLabel = numpy.argmax(prediction)
            actualLabel = testLabels[index][0]

            if predictedLabel == targetClass:
                if predictedLabel == actualLabel:
                    confusionMatrix['truePositive'] += 1
                else:
                    confusionMatrix['falsePositive'] += 1
            else:
                if predictedLabel == actualLabel:
                    confusionMatrix['trueNegative'] += 1
                else:
                    confusionMatrix['falseNegative'] += 1

            index += 1

        return confusionMatrix

newAgent = Agent()
newAgent.generateConfusionMatrix('superclass')