# Neural Network project
This is a simple neural network designed to classify images of objects and scenes at three different levels of detail and accuracy.

## Prerequisites

Tensorflow needs to be installed with pip or anaconda to run.

## Running the code

To train the models run trainModels.py, the hyperparameters are in hyperparameters.py and the model design is in model.py. Training takes about three hours with GPU acceleration.

The trained models are stored into the models folder and are loaded by the agent, but are too large to upload to github. To run some predictions, instantiate an Agent object from agent.py.