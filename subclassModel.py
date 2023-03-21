from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

cifar = keras.datasets.cifar100
(trainingData, trainingLabels), (testData, testLabels) = cifar.load_data(label_mode='fine')

cifar10FineLabels = [
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

model.fit(
    x=trainingData, 
    y=trainingLabels, 
    epochs=20,
    validation_split=0.2,
    shuffle=True
)

""" 
Latest Error Message:

File "/opt/homebrew/lib/python3.9/site-packages/keras/backend.py", line 5119, in categorical_crossentropy
        target.shape.assert_is_compatible_with(output.shape)

    ValueError: Shapes (32, 1) and (32, 100) are incompatible
""" 