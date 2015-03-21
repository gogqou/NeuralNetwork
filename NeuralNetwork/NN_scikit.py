'''
Created on Mar 20, 2015

@author: gogqou
'''

import numpy as np
import sys
import math
from Bio import SeqIO
from pylab import plot, show, ylim, yticks
import matplotlib.pyplot as plt
from matplotlib import *
import sklearn 
from sklearn import svm

import cluster_edited as cluster


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
# gradient descent method to calculate error and help out with backprop       #
def make_training_set(file, posorneg= 1):
    
    file_lines = readtxt(file)
    
    seqs = np.empty([68, len(file_lines)])
    
    seq_dict = {}
    sequenceList = []
    i = 0
    for file_line in file_lines:
        
        seq = Sequence(file_line, posorneg)
        seq_dict[seq.seq] = seq.label
        seqs[:,i] = np.array([seq.vector_rep[:,0]])
        sequenceList.append(seq)
        i = i+1
    return seqs, sequenceList, seq_dict
###############################################################################

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
    sample_indices = np.random.randint(0, len(nmer_seqs), [20000,1])
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
#  test training set with new sequences                                       #
def testoutput(test_set, test_output, sequenceList, outputFilename):
    
    out = open(outputFilename, 'w')
    for i in range(len(sequenceList)):
        out.write(sequenceList[i].seq+ '\t'+str(test_output[i])+ '\n')
    
    print 'done test and writing output file'
    return 1

###############################################################################
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
    #negseqs_string, neg_seq_dict = gen_nmers_from_fa(negative_fa_file, 17, directory, pos_dict)
    #this generates a entire set of negative sequences
    #from which we took a sample and put into sample_nseqs.txt
    
    #makes training set of the negative sequences
    
    negseqs, neg_sequenceList, neg_dict = make_training_set(directory + 'sample_nseqs_cluster.txt', 0)
    #puts the pos and neg sets together
    full_training_set = np.transpose(np.hstack((posseqs, negseqs)))
    full_sequenceList = pos_sequenceList + neg_sequenceList
    full_training_labels = np.zeros([len(full_sequenceList)])
    for i in range(len(full_sequenceList)):
        full_training_labels[i] = full_sequenceList[i].label
        
    clf = svm.SVR(kernel = 'sigmoid', verbose= True)
    clf.fit(full_training_set, full_training_labels) 
    
    
    test_output_file_name = 'sklearn_test_output.txt'
    testseqs, test_sequenceList, test_dict = make_training_set(test_file, 0.5)
    test_outputs =  clf.predict(np.transpose(testseqs))
    print test_outputs[1]
    testoutput(testseqs, test_outputs, test_sequenceList, directory+test_output_file_name)
    return 1

if __name__ == '__main__':
    main()