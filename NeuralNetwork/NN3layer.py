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
        self.bias = np.ones([1, self.inputshapey]) #always just set bias term to 1
        self.input = inputx        
        self.n_inputs = self.inputshapex #allows you to set how many inputs are in the first layer
        self.n_outputs = outputnum #allows you to set how many outputs come from the neural network
        self.n_hidden_nodes = hiddennodes
        self.Hlayer = np.zeros([hiddennodes+1, 1]) #add 1 to make space for the bias node
        self.weights_Hlayer = np.ones([self.n_inputs+1, hiddennodes+1]) 
        #add one to each dimension to add space for the bias nodes in input and hidden layers
        self.weights_output = np.ones([hiddennodes+1,self.n_outputs])
        
        #add one in vertical dimension to make space for bias node in hidden layer

        self.input = np.vstack((self.input, self.bias))
        #initialize the weights with random small numbers
        for i in range(self.n_inputs+1):
            for j in range(self.n_hidden_nodes+1):
                self.weights_Hlayer[i,j] = np.random.random()-np.random.random()
                
        for k in range(self.n_hidden_nodes+1):
            for m in range(self.n_outputs):
                self.weights_output[k,m] = np.random.random()-np.random.random()
             
    def forwardprop(self, x):
        #method to calculate new output value
        [self.inputshapex, self.inputshapey] = x.shape
        self.bias = np.ones([1, self.inputshapey]) #always just set bias term to 1
        self.input = x        
        self.n_inputs = self.inputshapex
        self.input = np.vstack((self.input, self.bias))
        
        #add on the bias node to input
        #set up transpose of the weights matrices for matrix multiplication
        weights_t = np.transpose(self.weights_Hlayer)
        weights_t_out = np.transpose(self.weights_output)
        #forward propagate using the weights and the values in each layer
        self.Hlayer = np.dot(weights_t, self.input)
        self.output = sigmoid(np.dot(weights_t_out, sigmoid(self.Hlayer)))
        print 'done forward prop'
              
        
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
                self.vector_rep[j] = 1
            elif self.seq[i] is 'T':
                self.vector_rep[j+1] = 1
            elif self.seq[i] is 'C':
                self.vector_rep[j+2] = 1
            elif self.seq[i] is 'G':
                self.vector_rep[j+3] = 1
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
        print i
        
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
    sample_indices = np.random.randint(0, len(nmer_seqs), [2000,1])
    training_sample = nmer_seqs[sample_indices]
    writetxt2(training_sample, directory + 'sample_nseqs.txt' )
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
# calc error for training                                                     #
def cost_func(NeuralNetwork, training_set, sequenceList):
    print 'calculating cost_func'
    [x,y] = training_set.shape
    labels = np.zeros([y,1])
    for i in range(y):
        labels[i]=sequenceList[i].label
    NeuralNetwork.forwardprop(training_set)
    print NeuralNetwork.output.shape
    #error = 1/2(y-f(x))^2
    errors = .5*np.square(np.transpose(NeuralNetwork.output)-labels)
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
    positive_sequences_file = sys.argv[1]
    negative_fa_file = sys.argv[2]
    directory = '/home/gogqou/Documents/Classes/bmi203-final-project/'
    posseqs, pos_sequenceList, pos_dict = make_training_set(positive_sequences_file)
    #negseqs_string, neg_seq_dict = gen_nmers_from_fa(negative_fa_file, 17, directory, pos_dict)
    negseqs, neg_sequenceList, neg_dict = make_training_set(directory + 'sample_nseqs.txt', 0)
    #print negseqs.shape
    NN = Network(posseqs,1,10,'sigmoid')
    NN.forwardprop(posseqs)
    errors_pos = cost_func(NN, posseqs, pos_sequenceList)
    print errors_pos
    
    
    
    NN.forwardprop(negseqs)
    errors_neg = cost_func(NN, negseqs, neg_sequenceList)
    print errors_neg
    return 1

if __name__ == '__main__':
    main()