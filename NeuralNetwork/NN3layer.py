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
import cluster_edited as cluster
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
        
        self.outputz = np.dot(weights_t_out, np.vstack((sigmoid(self.Hlayer), self.bias_Hlayer)))
        #print 'output'
        #print self.outputz
        self.output = sigmoid(self.outputz)
        #print 'done forward prop'
              
        
###############################################################################

###############################################################################
#                                                                             #
# class for a sequence                                                        #

class Sequence:

    def __init__(self, seq_ATCG, label):
        self.seq = seq_ATCG
        self.length = len(seq_ATCG)
        self.vector_rep = np.zeros([self.length*4, 1])
        self.vector_length = 4*len(self.vector_rep)
        self.nuc_ordering = ['A', 'T', 'C', 'G']
        self.label = label
        
        for i in range(self.length): 
            j = i*4
            if self.seq[i] is 'A':
                self.vector_rep[j] = 1.0
            elif self.seq[i] is 'T':
                self.vector_rep[j+1] = 1.0
            elif self.seq[i] is 'C':
                self.vector_rep[j+2] = 1.0
            elif self.seq[i] is 'G':
                self.vector_rep[j+3] = 1.0
            else:
                print 'nucleotide not found'
                break
###############################################################################
def sigmoid(x):
    return 1/(1+np.exp(-x))
###############################################################################
#                                                                             #
# gradient descent method to calculate error and help out with backprop       #
def make_training_set(file, posorneg= 1):
    
    file_lines = readtxt(file)
    seqs = np.zeros([68, 1])
    
    seq_dict = {}
    sequenceList = []
    i = 0
    for file_line in file_lines:
        seq = Sequence(file_line, posorneg)
        seq_dict[seq.seq] = seq.label
        seqs = np.hstack((seqs, seq.vector_rep))
        sequenceList.append(seq)
        i = i+1        
    seqs = seqs[:,1:i+1]
    return seqs, sequenceList, seq_dict
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
def writetxt3(seqList, filename):
    out = open(filename, 'w')
    for i in range(len(seqList)):
        out.write(seqList[i].seq+'\n')
    out.close()
    print 'done writing'
    return 1
###############################################################################
#                                                                             #
# generate negative sequences from fa file of yeast UTRs                      #
def gen_nmers_from_fa(file, n, directory, pos_dict):
    neg_seq_dict = {}
    nmer_seqs = np.empty([1,1])
    uniq_dict= {}
    for seq_record in SeqIO.parse(file, "fasta"):
        seq = str(seq_record.seq)
        nmers_list = nmers(seq, n, uniq_dict, pos_dict)
        nmer_seqs = np.append(nmer_seqs, nmers_list)
    #writetxt(nmer_seqs, directory+'neg_nmers.txt')
    sample_indices = np.random.randint(0, len(nmer_seqs), [40,1])
    training_sample = nmer_seqs[sample_indices]
    training_sample_centers = cluster.negative_centers(training_sample, directory)
    writetxt3(training_sample_centers, directory + 'sample_nseqs_cluster.txt' )
    return nmer_seqs, neg_seq_dict

###############################################################################

###############################################################################
#                                                                             #
# generate n-long sequences from sequence                                     #
def nmers(seq, n, uniq_dict, pos_dict):
    seq_list = []
    for i in range(len(seq)-17):
        if seq[i:i+17] in pos_dict.keys():
            continue
        elif seq[i:i+17] in uniq_dict.keys():
            continue
        else:
            seq_list.append(seq[i:i+17])
    return seq_list


###############################################################################
###############################################################################
#                                                                             #
# backpropagation to calculate new weights                                    #
def backprop(NN, learning_speed, regularization):
    
    NN.errors_output = -np.transpose(NN.labels)- NN.output* NN.output*(1-NN.output)
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
def cost_func(NN, training_set, sequenceList, regularization):
    #print 'calculating cost_func'
    [x,y] = training_set.shape
    labels = np.zeros([y,1])
    
    for i in range(y):
        labels[i]=sequenceList[i].label
        
    NN.forwardprop(training_set)
    NN.labels = labels
    
    #error = 1/2(y-f(x))^2
    NN.errors = .5*np.square(np.transpose(NN.output)-labels)
    
    sum_weights_Hlayer = np.sum(np.square(NN.weights_Hlayer))
    sum_weights_output = np.sum(np.square(NN.weights_output))
    
    training_set_sample_num = len(NN.errors)
    
    NN.avg_error = 1.0/training_set_sample_num * np.sum(NN.errors) + regularization/2.0*(sum_weights_Hlayer + sum_weights_output)
    #NN.avg_error = 1.0/training_set_sample_num * np.sum(errors)
    return NN
###############################################################################


###############################################################################
#                                                                             #
#  train neural network with training set                                     #
def train_NN(NN, training_set, sequenceList, test_set, test_sequenceList, regularization, learning_speed, error_tolerance):
    
    NN.forwardprop(training_set)
    NN= cost_func(NN, training_set, sequenceList, regularization)
    error_change = 5
    last_error = 0
    test_error = 5
    last_test_error = 0
    
    weights_Hlayer = np.reshape(NN.weights_Hlayer, [680, 1])
    weights_output = np.reshape(NN.weights_output, [10, 1])
    errors_output = np.transpose(np.array([NN.errors[:,0]]))
    bias_Hlayer =NN.bias_weights_Hlayer
    i = 0
    
    while NN.avg_error> error_tolerance and test_error>=1e-9:
        print 'ITERATION = =============================================== ', i
        #forward propagate with the entire training set
        print 'error= ', NN.avg_error
        NN=backprop(NN, learning_speed, regularization)
        NN = cost_func(NN, training_set, sequenceList, regularization)
        
        
        weights_Hlayer = np.hstack((weights_Hlayer, np.reshape(NN.weights_Hlayer, [680, 1])))
        weights_output = np.hstack((weights_output, np.reshape(NN.weights_output, [10, 1])))
        errors_output= np.hstack((errors_output,np.transpose(np.array([NN.errors[:,0]]))))
        bias_Hlayer = np.hstack((bias_Hlayer, NN.bias_weights_Hlayer))
        
        
        error_change = last_error - NN.avg_error
        last_error = NN.avg_error
        
        NN2 = cost_func(NN, training_set, sequenceList, regularization)
        test_error = np.abs(last_test_error - NN2.avg_error) 
        last_test_error = NN2.avg_error
        error_temp = last_test_error - NN2.avg_error
        i = i+1
        if i>100 and error_change <=0:
            print 'error going wrong way, stopped early'
            break
        
    indices = np.linspace(0,i+1, i+1)
    plt.figure(1)
    plt.subplot(211)
    for m in range(1):
        plot(indices, errors_output[m,:], 'b')
    plt.subplot(212)
    for j in range(680):
        plot(indices, weights_Hlayer[j,:], 'k') 
        
    for k in range(10):
        plot(indices, weights_output[k,:], 'y') 
        #plot(indices, bias_Hlayer[k,:], '^') 
        #ylim([-1,1])
    show()
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
    positive_sequences_file = sys.argv[1]
    negative_fa_file = sys.argv[2]
    test_file = sys.argv[3]
    directory = '/home/gogqou/Documents/Classes/bmi203-final-project/'
    
    #makes training set of the positive sequences
    posseqs, pos_sequenceList, pos_dict = make_training_set(positive_sequences_file, 1)
    
    #only need to do this once:
    negseqs_string, neg_seq_dict = gen_nmers_from_fa(negative_fa_file, 17, directory, pos_dict)
    #this generates a entire set of negative sequences
    #from which we took a sample and put into sample_nseqs.txt
    
    #makes training set of the negative sequences
    
    #negseqs, neg_sequenceList, neg_dict = make_training_set(directory + 'sample_nseqs.txt', 0)
    
    #puts the pos and neg sets together
    #full_training_set = np.hstack((posseqs, negseqs))
    #full_sequenceList = pos_sequenceList + neg_sequenceList
    #initiate the neural network
    #NN = Network(posseqs,1,10,'sigmoid')
    '''
    currentcost = 100
    best_reg = .23
    learning_speed = .1
    error_tolerance = 1e-5
    

    test_output_file_name = 'test_output.txt'
    testseqs, test_sequenceList, test_dict = make_training_set(test_file, 0.5)
    
    NN= train_NN(NN, full_training_set, full_sequenceList, testseqs, test_sequenceList, best_reg, learning_speed = .1, error_tolerance = 1e-6)
    
    print best_reg
    test_NN(NN, testseqs, test_sequenceList, directory + test_output_file_name)
    '''
    return 1

if __name__ == '__main__':
    main()