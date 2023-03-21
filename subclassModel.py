import time
from datetime import timedelta

from keras.applications.inception_v3 import (decode_predictions,
                                             preprocess_input)
from tensorflow import keras
from tensorflow import config
from tensorflow.keras.applications.inception_v3 import InceptionV3

if (len(config.list_physical_devices('GPU')) > 0):
    print('Using GPU')
else:
    print('Using CPU')


hyperparameters = {
    'learningRate': 0,
    'validationSplit': 0,
    'trainableInceptionLayers': 0,
    'epochs': 0,
}

cifar = keras.datasets.cifar100
(trainingData, trainingLabels), (testData, testLabels) = cifar.load_data(label_mode='fine')

cifar100FineLabels: list[str] = [
                    'beaver', 'dolphin', 'otter', 'seal', 'whale',
                    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                    'bottles', 'bowls', 'cans', 'cups', 'plates',
                    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                    'bed', 'chair', 'couch', 'table', 'wardrobe',
                    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                    'bear', 'leopard', 'lion', 'tiger', 'wolf',
                    'bridge', 'castle', 'house', 'road', 'skyscraper',
                    'cloud', 'forest', 'mountain', 'plain', 'sea',
                    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                    'crab', 'lobster', 'snail', 'spider', 'worm',
                    'baby', 'boy', 'girl', 'man', 'woman',
                    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                    'maple', 'oak', 'palm', 'pine', 'willow',
                    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
                ]

inputLayer = keras.Input(shape=(32, 32, 3)) # We need to create an input layer that matches our image size
# The pretrained models are setup with larger image inputs, so we need to preprocess to resize
resizeLayer = keras.layers.Resizing(299, 299)(inputLayer)
baseModel = InceptionV3()
for layer in baseModel.layers[:-50]:
    layer.trainable = False # Freeze lower original weights to prevent overfitting
pretrainedLayer = baseModel(resizeLayer)
outputLayer = keras.layers.Dense(100, activation='softmax')(pretrainedLayer)

model = keras.Model(inputLayer, outputLayer)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

start = time.perf_counter()
model.fit(
    x=trainingData, 
    y=keras.utils.to_categorical(trainingLabels, num_classes=len(cifar100FineLabels)), 
    epochs=20,
    validation_split=0.2,
    shuffle=True
)
end = time.perf_counter()
print(timedelta(seconds=end-start))