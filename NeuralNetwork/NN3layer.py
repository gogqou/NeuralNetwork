'''
Created on Mar 9, 2015

@author: gogqou



three layer neural network with n inputs and 1 output; 1 hidden layer with m nodes

take transcription factor binding sites as input and outputs a likelihood / rating of whether 
Rap1 will bind to this site


trained on positive results and negative Yeast sites


then tested against provided sites (don't want binary output so we can do ROC graph)

instrumented for cross validation
'''

import numpy as np
import sys
import math
###############################################################################
#                                                                             #
# class for a neural network                                                  #

class Network:

    def __init__(self, inputx, outputnum, hiddennodes, activation_func ):
        self.input = inputx
        self.n_inputs = len(self.input) #allows you to set how many inputs are in the first layer
        self.n_outputs = outputnum #allows you to set how many outputs come from the neural network
        self.n_hidden_nodes = hiddennodes
        self.bias = 1 #always just set bias term to 1
        self.Hlayer = np.zeros([hiddennodes+1, 1]) #add 1 to make space for the bias node
        self.weights_Hlayer = np.zeros([self.n_inputs+1, hiddennodes+1]) 
        #add one to each dimension to add space for the bias nodes in input and hidden layers
        self.weights_output = np.zeros([hiddennodes+1,self.n_outputs])
        
        #add one in vertical dimension to make space for bias node in hidden layer
        self.input = np.vstack((self.input, [self.bias]))
        #initialize the weights with random small numbers
        for i in range(self.n_inputs+1):
            for j in range(self.n_hidden_nodes+1):
                self.weights_Hlayer[i,j] = np.random.random()
                
        for k in range(self.n_hidden_nodes+1):
            for m in range(self.n_outputs):
                self.weights_output[k,m] = np.random.random()
    def forwardprop(self, x):
        #method to calculate new output value
        self.input = x
        self.n_inputs = len(self.input)
        self.input = np.vstack((self.input, [self.bias]))
        
        #add on the bias node to input
        #set up transpose of the weights matrices for matrix multiplication
        weights_t = np.transpose(self.weights_Hlayer)
        weights_t_out = np.transpose(self.weights_output)
        #forward propagate using the weights and the values in each layer
        self.Hlayer = np.dot(weights_t, self.input)
        self.output = np.dot(weights_t_out, self.Hlayer)
              
        
###############################################################################

###############################################################################
#                                                                             #
# class for a sequence                                                        #

class Sequence:

    def __init__(self, seq_ATCG):
        self.seq = seq_ATCG
        self.length = len(seq_ATCG)
        self.vector_rep = []
        self.vector_length = len(self.vector_rep)
        
    
        

###############################################################################
#                                                                             #
# calc error for training                                                     #
def cost_func(NeuralNetwork, training_set, t_set_value):
    [x,y] = training_set.shape
    training_set_outputs = t_set_value*np.ones([y, 1])
    NN_outputs = np.zeros([y, 1])
    
    errors =np.zeros([y, 1])
    for i in range(y):
        training_set_temp = np.matrix(training_set[:,i])
        NeuralNetwork.forwardprop(training_set_temp.T) 
        NN_outputs[i] = NeuralNetwork.output
        errors[i] = .5*math.pow((NN_outputs[i]-training_set_outputs[i]), 2)
    
    return errors
###############################################################################

###############################################################################
#                                                                             #
# backpropagation to calculate new weights                                    #
def backward_propagation(NeuralNetwork):
    print 1
    return NeuralNetwork
###############################################################################



###############################################################################
#                                                                             #
# gradient descent method to calculate error and help out with backprop       #
def grad_des(NeuralNetwork):
    
    
    
    return NeuralNetwork
###############################################################################


def main():
    np.set_printoptions(threshold=1000, linewidth=1000, precision = 5, suppress = False)
    
    
    temp_input = np.ones([4,1])
    temp_input[2,0]= 2
    NN = Network(temp_input,1,10,'sigmoid')
    NN.forwardprop(temp_input)
    
    training_set = np.hstack((temp_input,temp_input))
    errors = cost_func(NN, training_set, 1)
    print errors
    return 1

if __name__ == '__main__':
    main()