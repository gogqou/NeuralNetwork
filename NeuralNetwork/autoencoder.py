'''
Created on Mar 16, 2015


edits on top of NN3layer to do autoencoding


@author: gogqou
'''
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
from Bio import SeqIO
###############################################################################
#                                                                             #
# class for a neural network                                                  #

class Network:

    def __init__(self, inputx, outputnum, hiddennodes, activation_func ):
        [self.inputshapex, self.inputshapey] = inputx.shape
        self.input = inputx        
        self.n_inputs = self.inputshapex #allows you to set how many inputs are in the first layer
        self.n_outputs = outputnum #allows you to set how many outputs come from the neural network
        self.n_hidden_nodes = hiddennodes
        
        self.Hlayer = np.zeros([hiddennodes, self.inputshapey]) 
        self.weights_Hlayer = np.ones([hiddennodes, self.n_inputs]) 
        self.weights_output = np.ones([self.n_outputs, hiddennodes])
        
        
        
        self.bias_input = np.ones([1, self.inputshapey]) #always just set bias term to 1
        self.bias_weights_Hlayer = np.zeros([self.n_hidden_nodes,1])
        self.bias_Hlayer = np.ones([1, self.inputshapey])
        self.bias_weights_output = np.zeros([self.n_outputs,1])
        
        #initialize the weights with random small numbers
        for i in range(self.n_hidden_nodes):
            for j in range(self.n_inputs):
                self.weights_Hlayer[i,j] = np.random.uniform(-1, 1)
                
        for k in range(self.n_outputs):
            for m in range(self.n_hidden_nodes):
                self.weights_output[k,m] = np.random.uniform(-1, 1)
        for p in range(self.n_hidden_nodes):
            self.bias_weights_Hlayer[p] = np.random.uniform(-1, 1)
        
        for q in range(self.n_outputs):
            self.bias_weights_output[q] = np.random.uniform(-1, 1)
             
    def forwardprop(self, x):
        #method to calculate new output value
        [self.inputshapex, self.inputshapey] = x.shape
        
        self.bias_input = np.ones([1, self.inputshapey]) #always just set bias term to 1
        self.bias_Hlayer = np.ones([1, self.inputshapey])
        self.input = x        
        self.n_inputs = self.inputshapex
        self.inputwithbias = np.vstack((self.input, self.bias_input))
        #add on the bias node to input
        #set up transpose of the weights matrices for matrix multiplication
        weights_t = np.hstack((self.weights_Hlayer, self.bias_weights_Hlayer))
        weights_t_out = np.hstack((self.weights_output, self.bias_weights_output))
        
        #forward propagate using the weights and the values in each layer
        self.Hlayer = np.dot(weights_t, self.inputwithbias)
        self.Hlayer_activation = sigmoid(self.Hlayer)
        '''
        print 'weights'
        print weights_t
        print weights_t_out
        print 'Hlayer'
        print self.Hlayer
        print self.Hlayer_activation
        '''
        self.outputz = np.dot(weights_t_out, np.vstack((self.Hlayer_activation, self.bias_Hlayer)))
        #print 'output'
        #print self.outputz
        self.output = sigmoid(self.outputz)
        print 'done forward prop'
              
        
###############################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

###############################################################################
def readtxt(filename):
    lines = [line.strip() for line in open(filename)]
    return lines
def writetxt(seqList, filename):
    out = open(filename, 'w')
    for i in range(len(seqList)):
        out.write(seqList[i]+'\n')
    out.close()
    print 'done writing'
    return 1
def writetxt2(seqList, filename):
    out = open(filename, 'w')
    for i in range(len(seqList)):
        out.write(seqList[i][0]+'\n')
    out.close()
    print 'done writing'
    return 1

###############################################################################
#                                                                             #
# backpropagation to calculate new weights                                    #
def backprop(NN, learning_speed, regularization):
    
    print 'outputs'
    print NN.output
    print NN.input
    print 'diff'
    print NN.input-NN.output
    print '1-output'
    
    print (1-NN.output)
    
    NN.errors_output = -(NN.input- NN.output)* NN.output*(1-NN.output)
    NN.errors_Hlayer = np.dot(np.transpose(NN.weights_output), NN.errors_output)* NN.Hlayer_activation*(1-NN.Hlayer_activation)
    
    print 'errors'
    print NN.errors_output
    print 'Hlayer'
    print NN.errors_Hlayer
    
    #initalizes delta matrices
    delta_weights_output = np.zeros([NN.n_outputs, NN.n_hidden_nodes])
    delta_weights_Hlayer = np.zeros([NN.n_hidden_nodes, NN.n_inputs])
    
    ##change in weights = empty matrices + gradient matrix of cost function (or error)
    ## delta W = delta W + grad W
    # grad W = error for next layer * transpose of activation of this layer
    delta_weights_output=delta_weights_output+np.dot(NN.errors_output, np.transpose(NN.Hlayer_activation))
    delta_weights_Hlayer= delta_weights_Hlayer+np.dot(NN.errors_Hlayer, np.transpose(NN.input))
    
    print 'weights'
    print NN.weights_output
    print 'Hlayer'
    print NN.weights_Hlayer
    
    print 'delta weights'
    print delta_weights_output
    print 'Hlayer'
    print delta_weights_Hlayer
    
    # for the biases, we have to multiply the error by the activation, which in this case is just 
    # a vector of 1s because bias is always 1
    delta_bias_weights_output = np.dot(NN.errors_output, np.transpose(NN.bias_Hlayer))
    delta_bias_weights_Hlayer = np.dot(NN.errors_Hlayer, np.transpose(NN.bias_input))
    
    # update weights
    # weights = weights - learning rate * ( average change in weights + regularization * current weights)
    NN.weights_output =NN.weights_output-learning_speed*( 1/NN.inputshapey * delta_weights_output + regularization* NN.weights_output)
    NN.bias_weights_output = NN.bias_weights_output-learning_speed*(1/NN.inputshapey * delta_bias_weights_output)
    #we don't apply regularization to the bias weights
    NN.weights_Hlayer = NN.weights_Hlayer- learning_speed*(1/NN.inputshapey * delta_weights_Hlayer + regularization* NN.weights_Hlayer)
    NN.bias_weights_Hlayer = NN.bias_weights_Hlayer -learning_speed*(1/NN.inputshapey * delta_bias_weights_Hlayer)

    print 'new weights'
    print NN.weights_output
    print 'Hlayer'
    print NN.weights_Hlayer
    return NN
###############################################################################


###############################################################################
#                                                                             #
# calc error for training                                                     #
def cost_func(NN, training_set, regularization):
    print 'calculating cost_func'
    [x,y] = training_set.shape
    NN.forwardprop(training_set)
    #error = 1/2(y-f(x))^2
    errors = .5*np.square(NN.output-NN.input)

    sum_weights_Hlayer = np.sum(np.square(NN.weights_Hlayer))
    sum_weights_output = np.sum(np.square(NN.weights_output))
    training_set_sample_num = len(errors)
    #NN.avg_error = 1/training_set_sample_num * np.sum(errors) + regularization/2*(sum_weights_Hlayer + sum_weights_output)
    NN.avg_error = 1.0/training_set_sample_num * np.sum(errors) 
    return NN
###############################################################################


###############################################################################
#                                                                             #
#  train neural network with training set                                     #
def train_NN(NN, training_set, learning_speed, error_tolerance):
    
    regularization = .3
    NN.forwardprop(training_set)
    NN= cost_func(NN, training_set, regularization)
    error_change = 5
    last_error = 0
    while NN.avg_error> error_tolerance and error_change>=1e-6:
        #forward propagate with the entire training set
        print 'error= ', NN.avg_error
        NN=backprop(NN, learning_speed, regularization)
        NN = cost_func(NN, training_set, regularization)
        error_change = np.abs(last_error - NN.avg_error)
        last_error = NN.avg_error
    return NN
###############################################################################

###############################################################################
#                                                                             #
#  test training set with new sequences                                       #
def test_NN(NN, test_set, sequenceList, outputFilename):
    NN.forwardprop(test_set)
    out = open(outputFilename, 'w')
    for i in range(len(sequenceList)):
        out.write(sequenceList[i].seq+ '\t'+str(NN.output[0][i])+ '\n')
    
    print 'done test and writing output file'
    return 1

###############################################################################

def main():
    np.set_printoptions(threshold=1000, linewidth=1000, precision = 5, suppress = False)

    directory = '/home/gogqou/Documents/Classes/bmi203-final-project/'
    inputs = np.random.randint(2, size = (4,3))
    #initiate the neural network
    NN = Network(inputs,4,3,'sigmoid')
    NN.forwardprop(inputs)
    NN= backprop(NN, learning_speed=.15, regularization=.3)
    NN.forwardprop(inputs)
    NN= backprop(NN, learning_speed=.15, regularization=.3)
    #NN= train_NN(NN, inputs, learning_speed = .15, error_tolerance = 1e-2)
    '''
    test_output_file_name = 'test_output.txt'
    testseqs, test_sequenceList, test_dict = make_training_set(test_file, 0.5)
    test_NN(NN, testseqs, test_sequenceList, directory + test_output_file_name)
    '''
    return 1

if __name__ == '__main__':
    main()