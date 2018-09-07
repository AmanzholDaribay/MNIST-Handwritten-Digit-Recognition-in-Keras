MNIST Handwritten Digit Recognition in Keras
Based on work of Gregor Koehler at https://nextjournal.com/gkoehler/digit-recognition-with-keras

AMANZHOL DARIBAY and DAUREN BINAZAROV - students at NAZARBAYEV UNIVERSITY

TO RUN: in command line type PYTHON ENV.PY

Network configuration:
 - Network Architecture is Keras Sequential Model
 - Input units are 784
 - Number of hidden layers are 2
 - Number of neurons (units) in hidden layers are 512
 - The differentiation for the training via backpropagation is happening behind without having to specified details
 - In order to prevent overfitting, it's been used Dropout which keeps some weights randomly assigned
 - Activation function in hidden layers is Relu
 - Output units are 10 and output layer function is Softmax
 - Loss function is Categorical Cross Entropy
 - Optimizer is Adam (stochastic gradient descent)
 - Metric is Accuracy
 - Iterations (epochs) are 20
 - Samples per update (batch size) is 128

Detailed explanation of the code is commented in env.py
