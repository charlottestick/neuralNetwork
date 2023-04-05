hyperparameters = {
    'learningRate': 0.1 ** 4,  # Raise to power of desired number of zeros including before decimal
    'epochs': 30,
    'validationSplit': 0.3,
    'frozenInceptionLayers': 311-20,  # Total layers is 311
    'transferLearningWeights': 'imagenet',  # 'imagenet' to transfer learn
    'lossFunction': 'sparse_categorical_crossentropy',
    'trainingMetrics': ['accuracy']
}

# Because CIFAR100 is reasonably large, we don't necessarily have to use transfer learning
# By setting random weights on inceptionv3 architecture and training all layers, we've reached accuracy of 0.82 and validation accuracy of 0.47 after only 10 epochs
# Possible over fitting as validation accuracy diverged after 3 epochs and started dropping after 9 epochs, maybe need to do some random image manipulation
# End result: it reached 0.53 validation accuracy, with 0.98 accuracy

# Training all layers takes longer per epoch, more weights to backpropagate through?
