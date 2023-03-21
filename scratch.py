from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

# Scratch file for experimenting

cifar = keras.datasets.cifar10
(trainingData, trainingLabels), (testData, testLabels) = cifar.load_data()
cifar10Labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
# This builtin loadData can only return one set of labels at a time, this is probably fine as each 
# model will probably be in a separate module or file or whatever

# My custom 'Organics' superclass can probably be derived be fetching the course labels and mapping 
# the labels to my labels

imageIndex = 20
image = preprocess_input(trainingData[imageIndex])
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

inputLayer = keras.Input(shape=(32, 32, 3))
resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
baseModel = InceptionV3()(resizeLayer)

model = keras.Model(inputLayer, baseModel)
prediction = model.predict(image)
# The labels returned from this are wrong but close because ImageNet has different classes, 
# this is fine as we haven't trained it yet or given it our classes
label = decode_predictions(prediction)[0][0]
print('Prediction made:', label[1], 'with', (label[2] * 100), 'percent confidence')
print('Actual label: ', cifar10Labels[trainingLabels[imageIndex][0]])