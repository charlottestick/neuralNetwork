from tensorflow import keras

# Scratch file for experimenting

cifar = keras.datasets.cifar100
(trainingData, trainingLabels), (testData, testLables) = cifar.load_data(label_mode="fine")
# This builtin loadData can only return one set of labels at a time, this is probably fine as each 
# model will probably be in a separate module or file or whatever

# My custom 'Organics' superclass can probably be derived be fetching the course labels and mapping 
# the labels to my labels
