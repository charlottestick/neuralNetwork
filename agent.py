import os
import random
from typing import Literal
import matplotlib.pyplot as plt

import numpy
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

from labels import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Agent:
    def __init__(self) -> None:
        try:
            self.superclassModel = load_model('./models/superclass')
            self.classModel = load_model('./models/class')
            self.subclassModel = load_model('./models/subclass')
        except:
            print("One or more model files weren't found, using InceptionV3 model")
            inputLayer = keras.Input(shape=(32, 32, 3))
            resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
            baseModel = InceptionV3()(resizeLayer)
            superclassOutputLayer = keras.layers.Dense(
                2, activation='softmax')(baseModel)
            classOutputLayer = keras.layers.Dense(
                20, activation='softmax')(baseModel)
            subclassOutputLayer = keras.layers.Dense(
                100, activation='softmax')(baseModel)

            try:
                self.superclassModel = load_model('./models/superclass')
            except:
                self.superclassModel = keras.Model(
                    inputLayer, superclassOutputLayer)
                print('Using InceptionV3 for superclass')

            try:
                self.classModel = load_model('./models/class')
            except:
                self.classModel = keras.Model(inputLayer, classOutputLayer)
                print('Using InceptionV3 for class')

            try:
                self.subclassModel = load_model('./models/subclass')
            except:
                self.subclassModel = keras.Model(
                    inputLayer, subclassOutputLayer)
                print('Using InceptionV3 for subclass')

        cifar = keras.datasets.cifar100
        (_trainingData, _trainingLabels), (self.testDataFine,
                                           self.testLabelsFine) = cifar.load_data(label_mode='fine')
        (_trainingData, _trainingLabels), (self.testDataCoarse,
                                           self.testLabelsCoarse) = cifar.load_data(label_mode='coarse')
        self.testLabelsNatural = []
        for i in range(len(self.testLabelsCoarse)):
            if naturalClassesFromCoarse.count(self.testLabelsCoarse[i]):
                self.testLabelsNatural.append([0])
            else:
                self.testLabelsNatural.append([1])

        self.testLabelsNatural = numpy.array(self.testLabelsNatural)

    def getRandomTestImage(self) -> int:
        imageIndex = random.randint(0, len(self.testDataCoarse) - 1)
        return imageIndex

    def showRandomImage(self) -> None:
        imageIndex: int = self.getRandomTestImage()
        image = self.testDataCoarse[imageIndex]
        superclassLabel: str = cifar100NaturalLabels[self.testLabelsNatural[imageIndex][0]]
        classLabel: str = cifar100CoarseLabels[self.testLabelsCoarse[imageIndex][0]]
        # The problem is that fine and course data are not in the same order, so we can't run both on the same image easily
        subclassLabel: str = cifar100FineLabels[self.testLabelsFine[imageIndex][0]]

        print('Labels: ' + superclassLabel + ', ' +
              classLabel + ', ' + subclassLabel)
        plt.figure(figsize=(1, 1))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def predict(self, classType: Literal['superclass', 'class', 'subclass'], imageIndex: int | None = None) -> None:
        if classType != 'superclass' and classType != 'class' and classType != 'subclass':
            print('Invalid class type')
            return

        if imageIndex == None:
            imageIndex = self.getRandomTestImage()
        image = self.testDataFine[imageIndex] if classType == 'subclass' else self.testDataCoarse[imageIndex]
        imageForPrediction = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2])

        if classType == 'superclass':
            superclassPrediction = self.superclassModel(imageForPrediction)
            print('\nPredicted superclass:', cifar100NaturalLabels[numpy.argmax(superclassPrediction)], 'with', round(
                (numpy.amax(superclassPrediction) * 100), 1), 'percent confidence')
            print('Actual superclass:   ',
                  cifar100NaturalLabels[self.testLabelsNatural[imageIndex][0]])

        if classType == 'class':
            classPrediction = self.classModel(imageForPrediction)
            print('Predicted class:', cifar100CoarseLabels[numpy.argmax(classPrediction)], 'with', round(
                (numpy.amax(classPrediction) * 100), 1), 'percent confidence')
            print('Actual class:   ',
                  cifar100CoarseLabels[self.testLabelsCoarse[imageIndex][0]])

        if classType == 'subclass':
            subclassPrediction = self.subclassModel(imageForPrediction)
            print('Predicted subclass:', cifar100FineLabels[numpy.argmax(subclassPrediction)], 'with', round(
                (numpy.amax(subclassPrediction) * 100), 1), 'percent confidence')
            print('Actual subclass:   ',
                  cifar100FineLabels[self.testLabelsFine[imageIndex][0]])

        plt.figure(figsize=(1, 1))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def generateConfusionMatrix(self, classType: str) -> dict[str, str | int] | None:
        if classType == 'superclass':
            model = self.superclassModel
            labelValues = cifar100NaturalLabels
            testLabels = self.testLabelsNatural
            testData = self.testDataCoarse
        elif classType == 'class':
            model = self.classModel
            labelValues = cifar100CoarseLabels
            testLabels = self.testLabelsCoarse
            testData = self.testDataCoarse
        elif classType == 'subclass':
            model = self.subclassModel
            labelValues = cifar100FineLabels
            testLabels = self.testLabelsFine
            testData = self.testDataFine
        else:
            print('Invalid class type')
            return

        targetClass = random.randrange(0, len(labelValues))

        confusionMatrix = {'targetClass': labelValues[targetClass], 'truePositive': 0,
                           'falsePositive': 0, 'trueNegative': 0, 'falseNegative': 0}
        print('\nTarget class:', confusionMatrix['targetClass'])

        predictions = model.predict(testData, verbose='0')

        index = 0
        totalPredictions = len(testData)
        for prediction in predictions:
            print('\r ', index + 1, 'out of', totalPredictions, end='')

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

        print('\n')
        return confusionMatrix

    def getPrecision(self, confusionMatrix) -> int:
        precision: int = confusionMatrix['truePositive'] / (
            confusionMatrix['truePositive'] + confusionMatrix['falsePositive'])
        return precision

    def getRecall(self, confusionMatrix) -> int:
        recall: int = confusionMatrix['truePositive'] / (
            confusionMatrix['truePositive'] + confusionMatrix['falseNegative'])
        return recall

    def getTestAccuracy(self, classType):
        if classType == 'superclass':
            model = self.superclassModel
            testLabels = self.testLabelsNatural
            testData = self.testDataCoarse
        elif classType == 'class':
            model = self.classModel
            testLabels = self.testLabelsCoarse
            testData = self.testDataCoarse
        elif classType == 'subclass':
            model = self.subclassModel
            testLabels = self.testLabelsFine
            testData = self.testDataFine
        else:
            print('Invalid class type')
            return

        test_accuracy = model.evaluate(
            x=testData, y=testLabels, verbose='0')[1]
        print('\n', classType, 'test accuracy:',
              (test_accuracy * 100), 'percent')
        return test_accuracy
