#######This program is a machine learning algorithm for classifying hand-written digits.#######

MINIST dataset is used during training. Tensorflow is used for the framework. 



The program does the following:
1. Train a fully connected neural network with 2 hidden layers and an output layer. 
   The two hidden layers have relu activation function and the output layer has softmax activation function due to the fact that it's a multinomial classification problem. 
   The training process implemented mini-batch gradient descent with momentum. 
2. Train a fully connected neural network with 2 hidden layers and an output layer using adam optimizor 

3. Train a fully connected neural network with 2 hidden layers and an output layer using batch normalization to deal with the drastic changes during gradient updates. 

4. Train a convolutional neural network with adam optimizor 
   Relu activation is used in hidden layers and softmax is used in output layer. kernel_size is 5,5








To train the model, run the main.py file in Python. This will load the data, train the model and output visualizations. 




