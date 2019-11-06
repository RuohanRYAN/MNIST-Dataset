#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

import time

def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
#     inputs: Xs        tranining examples (d*n)
#             Ys        training labels   (c * n)
#             ii        the list/vector of indexes of the training example to compute the gradient with respect to
#             gamma     L2 regularization constant
#             W         parameters        (c * d)
#     output: returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
    d,n = Xs.shape
    c,_ = Ys.shape
    nSample = len(ii)
    Xsamp = np.reshape(Xs[:,ii],(d,nSample))
    Ysamp = np.reshape(Ys[:,ii],(c,nSample))
    sfmaxSum = np.sum(np.exp(np.dot(W,Xsamp)),axis = 0)
    sfmax = (np.exp(np.dot(W,Xsamp))/sfmaxSum)-Ysamp
    grad = np.zeros((c,d))
    for i in range(nSample):
        grad = grad+np.dot(np.reshape(sfmax[:,i],(c,1)),np.reshape(Xsamp[:,i],(1,d)))
    grad = grad/nSample + gamma*W
    return grad 

# compute the error of the classifier 
def multinomial_logreg_error(Xs, Ys, W):
#     inputs: Xs        training examples (d * n)
#             Ys        labels            (c * n)
#             W         parameters        (c * d)
#     returns   the model error as a percentage of incorrect labels
    d,n = Xs.shape
    c,_ = Ys.shape
    yTrain = np.argmax(np.dot(W,Xs),axis=0)
    yExpect = np.argmax(Ys,axis=0)
    nWrong = len(np.argwhere(yTrain - yExpect))
    return nWrong/n

# compute the cross-entropy loss of the classifier
def multinomial_logreg_loss(Xs, Ys, gamma, W):
#     inputs: Xs        examples          (d * n)
#             Ys        labels            (c * n)
#             gamma     L2 regularization constant
#             W         parameters        (c * d)
#     returns   the model cross-entropy loss
    d,n = Xs.shape
    c,_ = Ys.shape
    sfmaxSum = np.sum(np.exp(np.dot(W,Xs)),axis = 0)
    sfmax = np.exp(np.dot(W,Xs))/sfmaxSum
    loss = -np.sum(np.multiply(np.log(sfmax),Ys))/n + (gamma/2)*(np.linalg.norm(W)**2)
    return loss
# gradient descent
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
# inputs: Xs              training examples (d * n)
#         Ys              training labels   (c * n)
#         gamma           L2 regularization constant
#         W0              the initial value of the parameters (c * d)
#         alpha           step size/learning rate
#         num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
#         monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
# returns         a list of model parameters, one every "monitor_period" epochs
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    gradient = []
    ii = np.linspace(0,n-1,n,dtype = int)
    for i in range(num_epochs):
        W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii,gamma, W)
        if((i+1)%monitor_period==0):
            gradient.append(W)
    return gradient

# gradient descent with nesterov momentum

def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
# inputs: Xs              training examples (d * n)
#         Ys              training labels   (c * n)
#         gamma           L2 regularization constant
#         W0              the initial value of the parameters (c * d)
#         alpha           step size/learning rate
#         beta            momentum hyperparameter
#         num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
#         monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
        
# returns:         a list of model parameters, one every "monitor_period" epochs
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    V0 = W0
    gradient = []
    ii = np.linspace(0,n-1,n,dtype = int)
    for i in range(num_epochs):
        V = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
        W = V + beta*(V-V0)
        V0 = V
        if((i+1)%monitor_period==0):
            gradient.append(W)
    return gradient

# SGD: run stochastic gradient descent with minibatching and sequential sampling order
#

def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
# inputs: Xs              training examples (d * n)
#         Ys              training labels   (c * n)
#         gamma           L2 regularization constant
#         W0              the initial value of the parameters (c * d)
#         alpha           step size/learning rate
#         B               minibatch size
#         num_epochs      number of epochs (passes through the training set) to run
#         monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
        
# returns         a list of model parameters, one every "monitor_period" batches
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    gradient = []
    count = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            b = i*B
            rg = np.arange(b, b+B)
            ii = np.random.choice(rg,size = B, replace=False)
            W = W - alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            count+=1
            if(count % monitor_period==0):
                gradient.append(W)
    return gradient 
    
# SGD + Momentum: add momentum to the previous algorithm
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
# inputs: Xs              training examples (d * n)
#         Ys              training labels   (c * n)
#         gamma           L2 regularization constant
#         W0              the initial value of the parameters (c * d)
#         alpha           step size/learning rate
#         beta            momentum hyperparameter
#         B               minibatch size
#         num_epochs      number of epochs (passes through the training set) to run
#         monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    V = W0
    gradient = []
    count = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            b = i*B
            rg = np.arange(b, b+B)
            ii = np.random.choice(rg,size = B, replace=False)
            V = beta*V-alpha* multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            count+=1
            if(count % monitor_period==0):
                gradient.append(W)
    return gradient 

# Adam Optimizer
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
# inputs: Xs              training examples (d * n)
#         Ys              training labels   (c * n)
#         gamma           L2 regularization constant
#         W0              the initial value of the parameters (c * d)
#         alpha           step size/learning rate
#         rho1            first moment decay rate ρ1
#         rho2            second moment decay rate ρ2
#         B               minibatch size
#         eps             small factor used to prevent division by zero in update step
#         num_epochs      number of epochs (passes through the training set) to run
#         monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
    d,n = Xs.shape
    c,_ = Ys.shape
    W = W0
    # initialize r, s and timestep
    r, s, t = np.zeros((c,d)), np.zeros((c,d)), 0
    weights = []
    for k in range(num_epochs):
        for i in range(int(n/B)):
            t += 1
            # impelement sequential sampling
            b = i*B
            rg = np.arange(b, b+B)
            ii = np.random.choice(rg,size = B, replace=False)
            # the average gradient of stochastic samples
            grad = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            # accumulate and correct first and second moment estimate
            s = rho1*s + (1-rho1)*grad
            r = rho2*r + (1-rho2)*np.square(grad)
            s_hat = s / (1 - np.power(rho1,t))
            r_hat = r/ (1 - np.power(rho2,t))
            # update model
            W = W- np.multiply(alpha / np.sqrt(r + eps),s)
            if(t % monitor_period==0):
                weights.append(W)
    return weights

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d,n = Xs_tr.shape
    c,_ = Ys_tr.shape
#   specify parameters 
    gamma = 0.0001 # l2 regularization constant
    alpha = 0.2 #learning rate
    W0 = np.zeros((c,d)) # initial wright vector
    num_epochs = 100 # number of epochs
    monitor_period = 10 # number of iterations
    B = 600 # mini-batch size
    rho1 = 0.9 #first moment decay rate ρ1
    rho2 = 0.999# second moment decay rate ρ2
    eps = 10e-5 # episillon
############ training 
    sgd_result = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    alpha = 0.01
    sgd_adam = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period)
    x = range(len(sgd_result))
    sgd_train_error = [multinomial_logreg_error(Xs_tr, Ys_tr, sgd_result[i]) for i in range(len(sgd_result))]
    adam_train_error = [multinomial_logreg_error(Xs_tr, Ys_tr, sgd_adam[i]) for i in range(len(sgd_adam))]
    pyplot.figure(1);pyplot.plot(x,sgd_train_error,x,adam_train_error);pyplot.title('training error for SGD');pyplot.legend(('sgd','adam'));pyplot.savefig('training error for SGD');
############## testing 
    sgd_test_error = [multinomial_logreg_error(Xs_te, Ys_te, sgd_result[i]) for i in range(len(sgd_result))]
    adam_test_error = [multinomial_logreg_error(Xs_te, Ys_te, sgd_adam[i]) for i in range(len(sgd_adam))]
    pyplot.figure(2);pyplot.plot(x,sgd_test_error,x,adam_test_error);pyplot.title('testing error for SGD');pyplot.legend(('sgd','adam'));pyplot.savefig('testing error for SGD');
############## plot the training loss 
    sgd_loss_error = [multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, sgd_result[i]) for i in range(len(sgd_result))]
    adam_loss_error = [multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, sgd_adam[i]) for i in range(len(sgd_adam))]
    pyplot.figure(3);pyplot.plot(x,sgd_loss_error,x,adam_loss_error);pyplot.title('training loss for SGD');pyplot.legend(('sgd','adam'));pyplot.savefig('testing loss for SGD');
    print('training error for SGD:', sgd_train_error[-1])
    print('training error for adam:', adam_train_error[-1])
    print('testing error for SGD:', sgd_test_error[-1])
    print('testing error for adam:',adam_test_error[-1])
    print('training loss for SGD:',sgd_loss_error[-1])
    print('training loss for adam:',adam_loss_error[-1])
    

    

        
    
        