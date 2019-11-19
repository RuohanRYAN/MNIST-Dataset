#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import tensorflow as tf
mnist = tf.keras.datasets.mnist
import time 

### hyperparameter settings and other constants
batch_size = 128
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    mnist = tf.keras.datasets.mnist
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()
    Xs_tr = Xs_tr / 255.0
    Xs_te = Xs_te / 255.0
    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 28, 28, 1) # 28 rows, 28 columns, 1 channel
    Xs_te = Xs_te.reshape(Xs_te.shape[0], 28, 28, 1)
    return (Xs_tr, Ys_tr, Xs_te, Ys_te)



# evaluate a trained model on MNIST data, and print the usual output from TF
#
# Xs        examples to evaluate on
# Ys        labels to evaluate on
# model     trained model
#
# returns   tuple of (loss, accuracy)
def evaluate_model(Xs, Ys, model):
    (loss, accuracy) = model.evaluate(Xs, Ys)
    return (loss, accuracy)


# train a fully connected two-hidden-layer neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of 
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    # TODO students should implement this
    n,i,j,_ = Xs.shape
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(i, j,1)),
                               tf.keras.layers.Dense(d1, activation=tf.nn.relu),
                               tf.keras.layers.Dense(d2, activation=tf.nn.relu),
                               tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(tf.keras.optimizers.SGD(lr=alpha,momentum=beta),
                  loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(Xs,Ys,batch_size=B,epochs=epochs,validation_split=0.1)
    return [model,model.history]

# train a fully connected two-hidden-layer neural network on MNIST data using Adam, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of 
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, B, epochs):
    # TODO students should implement this
    n,i,j,_ = Xs.shape
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(i, j,1)),
                               tf.keras.layers.Dense(d1, activation=tf.nn.relu),
                               tf.keras.layers.Dense(d2, activation=tf.nn.relu),
                               tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(tf.keras.optimizers.Adam(lr=alpha,beta_1=rho1,beta_2=rho2),
                  loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(Xs,Ys,batch_size=B,epochs=epochs,validation_split=0.1)
    return [model,model.history]

# train a fully connected two-hidden-layer neural network with Batch Normalization on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of 
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    # TODO students should implement this
    n,i,j,_ = Xs.shape
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(i, j,1)))
    model.add(tf.keras.layers.Dense(d1));model.add(tf.keras.layers.BatchNormalization(momentum=0.9));
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(d2));model.add(tf.keras.layers.BatchNormalization(momentum=0.9));
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(10));model.add(tf.keras.layers.BatchNormalization(momentum=0.9));
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(tf.keras.optimizers.SGD(lr=alpha,momentum=beta),
                  loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(Xs,Ys,batch_size=B,epochs=epochs,validation_split=0.1)
    return [model,model.history]

# train a convolutional neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of 
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, epochs):
    # TODO students should implement this
    n,i,j,_ = Xs.shape
    model =  tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,kernel_size=(5,5),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(64,kernel_size=(5,5),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(lr=alpha,beta_1=rho1,beta_2=rho2),
                  loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(Xs,Ys,batch_size=B,epochs=epochs,validation_split=0.1)
    return [model,model.history]


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    print(Xs_tr.shape)
    print(Ys_tr.shape)
    SGD 
    d1 = 1024
    d2 = 256
    alpha = 0.1
    beta = 0
    B = 128
    epochs = 10
    start = time.time()
    model = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, B, epochs)
    end = time.time()
    floss,faccuracy = evaluate_model(Xs_te, Ys_te, model[0])
    print('final test loss and accuracy is:', floss,' and ', faccuracy)
    print('totle elapse time is:', (end-start))
#     print(model[1].history.keys())
    print(model[1].history['acc'])
    print(model[1].history['val_acc'])
    print(model[1].history['loss'])
    print(model[1].history['val_loss'])
    pyplot.figure(1)
    pyplot.plot(model[1].history['loss']);pyplot.plot(model[1].history['val_loss']);pyplot.plot([floss for i in range(epochs)])
    pyplot.title('loss vs the number of epochs for SGD');pyplot.ylabel('loss');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper left');pyplot.savefig('loss vs the number of epochs for SGD')
    pyplot.figure(2)
    pyplot.plot(model[1].history['acc']);pyplot.plot(model[1].history['val_acc']);pyplot.plot([faccuracy for i in range(epochs)])
    pyplot.title('accuracy vs the number of epochs for SGD');pyplot.ylabel('accuracy');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper right');pyplot.savefig('accuracy vs the number of epochs for SGD')
    
SGD with momentum 
    d1 = 1024
    d2 = 256
    alpha = 0.1
    beta = 0.9
    B = 128
    epochs = 10
    start = time.time()
    model = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, B, epochs)
    end = time.time()
    floss,faccuracy = evaluate_model(Xs_te, Ys_te, model[0])
    print('final test loss and accuracy is:', floss,' and ', faccuracy)
    print('totle elapse time is:', (end-start))
#     print(model[1].history.keys())
    print(model[1].history['acc'])
    print(model[1].history['val_acc'])
    print(model[1].history['loss'])
    print(model[1].history['val_loss'])
    pyplot.figure(3)
    pyplot.plot(model[1].history['loss']);pyplot.plot(model[1].history['val_loss']);pyplot.plot([floss for i in range(epochs)])
    pyplot.title('loss vs the number of epochs for SGD with momentum');pyplot.ylabel('loss');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper left');pyplot.savefig('loss vs the number of epochs for SGD with momentum')
    pyplot.figure(4)
    pyplot.plot(model[1].history['acc']);pyplot.plot(model[1].history['val_acc']);pyplot.plot([faccuracy for i in range(epochs)])
    pyplot.title('accuracy vs the number of epochs for SGD with momentum');pyplot.ylabel('accuracy');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper right');pyplot.savefig('accuracy vs the number of epochs for SGD with momentum')
    ADAM
    d1 = 1024
    d2 = 256
    alpha = 0.001
    rho1 = 0.99
    rho2 = 0.999
    B = 128
    epochs = 10
    start = time.time()
    model = train_fully_connected_adam(Xs_tr, Ys_tr, d1, d2, alpha, rho1, rho2, B, epochs)
    end = time.time()
    floss,faccuracy = evaluate_model(Xs_te, Ys_te, model[0])
    print('final test loss and accuracy is:', floss,' and ', faccuracy)
    print('totle elapse time is:', (end-start))
    print(model[1].history['acc'])
    print(model[1].history['val_acc'])
    print(model[1].history['loss'])
    print(model[1].history['val_loss'])
    pyplot.figure(5)
    pyplot.plot(model[1].history['loss']);pyplot.plot(model[1].history['val_loss']);pyplot.plot([floss for i in range(epochs)])
    pyplot.title('loss vs the number of epochs with Adam');pyplot.ylabel('loss');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper left');pyplot.savefig('loss vs the number of epochs for Adam')
    pyplot.figure(6)
    pyplot.plot(model[1].history['acc']);pyplot.plot(model[1].history['val_acc']);pyplot.plot([faccuracy for i in range(epochs)])
    pyplot.title('accuracy vs the number of epochs with Adam');pyplot.ylabel('accuracy');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper right');pyplot.savefig('accuracy vs the number of epochs with Adam')
    Batch normalization 
    d1 = 1024
    d2 = 256
    alpha = 0.001
    beta = 0.9
    B = 128
    epochs = 10
    start = time.time()
    model = train_fully_connected_bn_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, B, epochs)
    end = time.time()
    floss,faccuracy = evaluate_model(Xs_te, Ys_te, model[0])
    print('final test loss and accuracy is:', floss,' and ', faccuracy)
    print('totle elapse time is:', (end-start))
    print(model[1].history['acc'])
    print(model[1].history['val_acc'])
    print(model[1].history['loss'])
    print(model[1].history['val_loss'])
    pyplot.figure(7)
    pyplot.plot(model[1].history['loss']);pyplot.plot(model[1].history['val_loss']);pyplot.plot([floss for i in range(epochs)])
    pyplot.title('loss vs the number of epochs with batchnormalization');pyplot.ylabel('loss');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper left');pyplot.savefig('loss vs the number of epochs with batchnormalization')
    pyplot.figure(8)
    pyplot.plot(model[1].history['acc']);pyplot.plot(model[1].history['val_acc']);pyplot.plot([faccuracy for i in range(epochs)])
    pyplot.title('accuracy vs the number of epochs with batchnormalization');pyplot.ylabel('accuracy');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper right');pyplot.savefig('accuracy vs the number of epochs with batchnormalization')

   CNN 
    d1 = 1024
    d2 = 256
    alpha = 0.001
    rho1=0.99
    rho2=0.999
    B = 128
    epochs = 10
    start = time.time()
    model = train_CNN_sgd(Xs_tr, Ys_tr, alpha, rho1, rho2, B, epochs)
    end = time.time()
    floss,faccuracy = evaluate_model(Xs_te, Ys_te, model[0])
    print('final test loss and accuracy is:', floss,' and ', faccuracy)
    print('totle elapse time is:', (end-start))
    print(model[1].history['acc'])
    print(model[1].history['val_acc'])
    print(model[1].history['loss'])
    print(model[1].history['val_loss'])
    pyplot.figure(9)
    pyplot.plot(model[1].history['loss']);pyplot.plot(model[1].history['val_loss']);pyplot.plot([floss for i in range(epochs)])
    pyplot.title('loss vs the number of epochs with CNN and ADAM');pyplot.ylabel('loss');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper left');pyplot.savefig('loss vs the number of epochs with CNN and ADAM')
    pyplot.figure(10)
    pyplot.plot(model[1].history['acc']);pyplot.plot(model[1].history['val_acc']);pyplot.plot([faccuracy for i in range(epochs)])
    pyplot.title('accuracy vs the number of epochs with CNN and ADAM');pyplot.ylabel('accuracy');pyplot.xlabel('epochs');pyplot.legend(['train', 'validation','test'], loc='upper right');pyplot.savefig('accuracy vs the number of epochs with CNN and ADAM')
    
    
