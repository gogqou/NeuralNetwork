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
from pylab import plot, show, ylim, yticks
import matplotlib.pyplot as plt
from matplotlib import *
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

        self.outputz = np.dot(weights_t_out, np.vstack((self.Hlayer_activation, self.bias_Hlayer)))
        #print 'output'
        self.output = sigmoid(self.outputz)
        
        #print self.output
        #print 'done forward prop'
              
        
###############################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))
def threshold(x):
    if x >.5:
        return 1
    elif x<=.5:
        return 0
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
    NN.errors_output = -(NN.input- NN.output)* NN.output*(1-NN.output)
    NN.errors_Hlayer = np.dot(np.transpose(NN.weights_output), NN.errors_output)* NN.Hlayer_activation*(1-NN.Hlayer_activation)
    
  
    
    #initalizes delta matrices
    delta_weights_output = np.zeros([NN.n_outputs, NN.n_hidden_nodes])
    delta_weights_Hlayer = np.zeros([NN.n_hidden_nodes, NN.n_inputs])
    
    ##change in weights = empty matrices + gradient matrix of cost function (or error)
    ## delta W = delta W + grad W
    # grad W = error for next layer * transpose of activation of this layer
    delta_weights_output=delta_weights_output+np.dot(NN.errors_output, np.transpose(NN.Hlayer_activation))
    delta_weights_Hlayer= delta_weights_Hlayer+np.dot(NN.errors_Hlayer, np.transpose(NN.input))
  
    
    # for the biases, we have to multiply the error by the activation, which in this case is just 
    # a vector of 1s because bias is always 1
    delta_bias_weights_output = np.dot(NN.errors_output, np.transpose(NN.bias_Hlayer))
    delta_bias_weights_Hlayer = np.dot(NN.errors_Hlayer, np.transpose(NN.bias_input))
    
    # update weights
    # weights = weights - learning rate * ( average change in weights + regularization * current weights)
    NN.weights_output =NN.weights_output-learning_speed*( 1.0/NN.inputshapey * delta_weights_output + regularization* NN.weights_output)
    NN.bias_weights_output = NN.bias_weights_output-learning_speed*(1.0/NN.inputshapey * delta_bias_weights_output)
    #we don't apply regularization to the bias weights
    NN.weights_Hlayer = NN.weights_Hlayer- learning_speed*(1.0/NN.inputshapey * delta_weights_Hlayer + regularization* NN.weights_Hlayer)
    NN.bias_weights_Hlayer = NN.bias_weights_Hlayer -learning_speed*(1.0/NN.inputshapey * delta_bias_weights_Hlayer)

    return NN
###############################################################################


###############################################################################
#                                                                             #
# calc error for training                                                     #
def cost_func(NN, training_set, regularization):
    #print 'calculating cost_func'
    NN.forwardprop(training_set)
    #error = 1/2(y-f(x))^2
    NN.errors = .5*np.square(NN.output-NN.input)
    
    sum_weights_Hlayer = np.sum(np.square(NN.weights_Hlayer))
    sum_weights_output = np.sum(np.square(NN.weights_output))
    training_set_sample_num = float(len(NN.errors))
    #output = NN.output
    #low_values_indices = output < .5  # Where values are low
    #output[low_values_indices] = 0 
    #high_val_indices = output>=.5
    #output[high_val_indices] = 1
    #NN.abs_error = 1.0/training_set_sample_num*np.sum(np.abs(NN.input-output))+ regularization/2.0*(sum_weights_Hlayer + sum_weights_output)
    NN.avg_error = 1.0/training_set_sample_num * np.sum(NN.errors) + regularization/2.0*(sum_weights_Hlayer + sum_weights_output)
    #NN.avg_error = 1.0/training_set_sample_num * np.sum(errors) 
    return NN
###############################################################################


###############################################################################
#                                                                             #
#  train neural network with training set                                     #
def train_NN(NN, training_set, test_set, learning_speed, error_tolerance, regularization = .008):
    NN.forwardprop(training_set)
    NN= cost_func(NN, training_set, regularization)
    error_change = 5
    last_error = 0
    test_error = 5
    last_test_error = 0
    
    weights_Hlayer = np.reshape(NN.weights_Hlayer, [24, 1])
    weights_output = np.reshape(NN.weights_output, [24, 1])
    errors_output = np.transpose(np.array([NN.errors[:,0]]))
    bias_Hlayer =NN.bias_weights_Hlayer
    #print NN.Hlayer_activation
    i=0
    while NN.avg_error> error_tolerance and test_error>=1e-6 and i <=900000:
        print 'ITERATION = =============================================== ', i
        #forward propagate with the entire training set
        #print 'error= ', NN.avg_error
        NN=backprop(NN, learning_speed, regularization)
        NN = cost_func(NN, training_set, regularization)
        
        weights_Hlayer = np.hstack((weights_Hlayer, np.reshape(NN.weights_Hlayer, [24, 1])))
        weights_output = np.hstack((weights_output, np.reshape(NN.weights_output, [24, 1])))
        errors_output= np.hstack((errors_output,np.transpose(np.array([NN.errors[:,0]]))))
        bias_Hlayer = np.hstack((bias_Hlayer, NN.bias_weights_Hlayer))
        
        error_change = np.abs(last_error - NN.avg_error)
        last_error = NN.avg_error
        
        NN2=cost_func(NN, test_set, regularization)
        test_error = np.abs(last_test_error - NN2.avg_error) 
        last_test_error = NN2.avg_error
        print 'error= ', NN2.avg_error
        i=i+1
    indices = np.linspace(0,i+1, i+1)
    
    plt.figure(1)
    plt.subplot(211)
    for m in range(8):
        plot(indices, errors_output[m,:], 'b')
    plt.subplot(212)
    for j in range(24):
        plot(indices, weights_Hlayer[j,:], 'k') 
        #plot(indices, weights_output[j,:], 'o') 
    #for k in range(3):
        #plot(indices, bias_Hlayer[k,:], '^') 
        #ylim([-1,1])
    #show()
    plt.savefig('/home/gogqou/Documents/Classes/bmi203-final-project/'+'learning_speed_'+str(learning_speed)+'_reg_'+str(regularization)+'run1.png')
    plt.clf()
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
    inputs = np.random.randint(2, size = (8,500)).astype(float)
    test_set = np.random.randint(2, size = (8,400)).astype(float)
    '''
    inputs = np.zeros([8,8])
    for i in range(8):
        inputs[i,i] = 1.0
    '''
    #initiate the neural network
    
    NN = Network(inputs,8,3,'sigmoid')
    currentcost = 100
    best_reg = .0015
    learning_speed = .4
    error_tolerance = 1e-3
    #regularization = np.linspace(0.0, .99, 1000)
    '''
    for j in range(len(regularization)):
        NN = Network(inputs,8,3,'sigmoid')
        for i in range(1,25):
            
            NN.forwardprop(inputs)
            NN= backprop(NN, learning_speed, regularization[j])
        NN = cost_func(NN, inputs, regularization[j])
        #print NN.avg_error
        if NN.avg_error < currentcost:
            currentcost = NN.avg_error
            best_reg = regularization[j]
    regularization = np.linspace(0.0, best_reg+best_reg*1.5, 500)
    for j in range(len(regularization)):
        NN = Network(inputs,8,3,'sigmoid')
        for i in range(1,25):
            NN.forwardprop(inputs)
            NN= backprop(NN, learning_speed, regularization[j])
        NN = cost_func(NN, inputs, regularization[j])
        #print NN.avg_error
        if NN.avg_error < currentcost:
            currentcost = NN.avg_error
            best_reg = regularization[j]
    regularization = np.linspace(0.0, best_reg+best_reg*1.5, 300)
    for j in range(len(regularization)):
        NN = Network(inputs,8,3,'sigmoid')
        for i in range(1,25):
            NN.forwardprop(inputs)
            NN= backprop(NN, learning_speed, regularization[j])
        NN = cost_func(NN, inputs, regularization[j])
        #print NN.avg_error
        if NN.avg_error < currentcost:
            currentcost = NN.avg_error
            best_reg = regularization[j]
    '''
    '''
    summary = np.zeros([1,3])
    for best_reg in range(3, 120, 8):
        best_reg = float(best_reg)/10000.0
        for learning_speed in range(1, 100,2):
            learning_speed = float(learning_speed/100.0)
            NN = Network(inputs,8,3,'sigmoid')        
            NN= train_NN(NN, inputs, test_set, float(learning_speed), error_tolerance, best_reg)
            summary = np.vstack((summary, [best_reg, learning_speed, NN.avg_error]))
            print 'reg= ', best_reg, 'learning speed = ', learning_speed, 'error = ', NN.avg_error 
    
    
    outputFilename =  directory+'summary_0to10.txt'
    out = open(outputFilename, 'w')
    for i in range(len(summary)):
        out.write('regularization  =' + str(summary[i][0])+ '\t'+ 'learning speed =  '+ str(summary[i][1])+ '\t' + 'error =  '+ str(summary[i][2])+ '\n')
    out.close()
    '''
    NN = Network(inputs,8,3,'sigmoid')        
    NN= train_NN(NN, inputs, test_set, float(learning_speed), error_tolerance, best_reg)
    NN.forwardprop(inputs)
    print NN.input
    output = NN.output
    low_values_indices = output < .5  # Where values are low
    output[low_values_indices] = 0 
    high_val_indices = output>=.5
    output[high_val_indices] = 1
    print output
    print NN.input-output
    print np.sum(np.abs(NN.input-output))
    '''
    test_output_file_name = 'test_output.txt'
    testseqs, test_sequenceList, test_dict = make_training_set(test_file, 0.5)
    test_NN(NN, testseqs, test_sequenceList, directory + test_output_file_name)
    '''
    return 1

if __name__ == '__main__':
    main()