'''
Created on Mar 19, 2015

@author: gogqou
'''
#from test.test_imageop import AAAAA
'''
Created on Feb 16, 2015

@author: gogqou

modified Scott Pegg's initial version


clustering algorithms:


hierarchical: 
bottom up

partitioning:
k mean 
'''

###############################################################################
###############################################################################
##                                                                           ##
##  MIS-203                                                                  ##
##  Clustering Lecture   (April 12, 2005)                                    ##
##  Programming Assignment                                                   ##
##                                                                           ##
##  Author: Scott Pegg                                                       ##
##                                                                           ##
##                                                                           ##
##  The abbreviated instructions:                                            ##
##    You will be given a set of enzyme active sites in PDB format           ##
##    (1) Implement a similarity metric                                      ##
##    (2) Implement a clustering method based on a partitioning algorithm    ##
##    (3) Implement a clustering method based on a hierarchical algorithm    ##
##    (4) Answer the questions given in the homework assignment              ##
##                                                                           ##
##  Please read the full instructions from the course website _before_ you   ##
##  start this assignment!                                                   ##
##                                                                           ##
###############################################################################
###############################################################################

import scipy.spatial.distance as distance

from string import *
from math import *
import sys, os
import glob 
import random
import numpy as np
import math
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt
import matplotlib



###############################################################################
#                                                                             #
# class for a sequence                                                        #

class Sequence:

    def __init__(self, seq_ATCG):
        self.seq = seq_ATCG
        self.length = len(seq_ATCG)
        self.vector_rep = np.zeros([np.power(3,4), 1])
        self.vector_length = len(self.vector_rep)
        self.center = []
        nucls = ['A', 'C', 'G', 'T']
        kmer_list = []
        self.kmer_dict = {}
        index = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    kmer_list.append(nucls[i]+nucls[j]+nucls[k])
                    self.kmer_dict[nucls[i]+nucls[j]+nucls[k]]= index
                    index = index+1          
        
    def nmers(self, k):
        for i in range(0, self.length-k):
            vector_index = self.kmer_dict[self.seq[i:i+3]]
            self.vector_rep[vector_index] = self.vector_rep[vector_index]+1
            
###############################################################################
#                                                                             #
# read in sequences
def read_sequences(file):
    file_lines = readtxt(file)   
    sequenceList = []
    for file_line in file_lines:
        seq = Sequence(file_line)
        seq.nmers()
        sequenceList.append(seq)



###############################################################################
def readtxt(filename):
    lines = [line.strip() for line in open(filename)]
    return lines                
###############################################################################
#                                                                             #
# Compute the similarity between two given ActiveSite instances.              #
#                                                                             #
# Input: two ActiveSite instances                                             #
# Output: the similarity between them (a floating point number)               #
#                                                                             #

def compute_similarity(seqA, seqB, tanimoto_dict):
    #similarity = tanimoto_dict[(site_A.name, site_B.name)]+distance.euclidean(site_A.metric, site_B.metric)
    similarity = distance.cityblock(seqA.metric, seqB.metric)
    return similarity

#                                                                             #
#                                                                             #
###############################################################################


########################################################################################################################################
#                                                                             #
# Cluster a given set of ActiveSite instances using a partitioning method.    #
#                                                                             #
# Input: a list of sequences instances                                       #
# Output: a clustering of sequences instances                                #
#         (this is really a list of clusters, each of which is list of        #
#         sequences instances)                                               #

def cluster_by_partitioning(sequences):
    
    # randomly pick k centers
    #iterate through k and keep the lowest objective function seqs
    best_obj_func = 10000
    #go through different values for number of seqs to find the one that gives the lowest obj function
    for k in range(1,len(sequences)/4, 3):
        #do a few tries at randomly generating centers
        for repeat in range(4):
            centers_indices = random.sample(xrange(0, len(sequences)), k)
            centers = []
            for i in range(len(centers_indices)):
                centers.append(sequences[centers_indices[i]])
            #centers just gives you the indices, not the actual instances
            #do k-means_seqs--this function takes the centers and clusters, calculates new clusters
            # k_means_centers takes new clusterings and calculates new centers
            current_clusters= k_means_seqs(sequences, orig_clusters, centers)
            previous_epsilon = obj_function( current_clusters, centers)
            L = 0
            delta = previous_epsilon
            while delta>10 and L<600:
                current_seqs= k_means_seqs(sequences, current_clusters, centers)
                current_obj_func = obj_function(current_clusters, centers)
                centers = k_means_centers(current_clusters, centers)
                epsilon = current_obj_func-obj_function(current_clusters, centers)
                delta = abs(epsilon - previous_epsilon)
                previous_epsilon = epsilon
                print 'clustering iteration', L, 'k = ', k
                L = L+1
            if current_obj_func <best_obj_func:
                best_clusters = current_clusters
                best_obj_func = current_obj_func
                best_k = k
            print current_obj_func, k
    print best_k
    print best_obj_func
    return best_clusters
#                                                                             #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
#                                                                             #
def obj_function(clusters, centers, tanimoto_dict):
    
    distance_matrix = np.zeros([len(centers),136])
    max_length = 0
    for i in range(len(centers)):
        for j in range(len(clusters[i])):
            distance_matrix[i,j] = math.pow(compute_similarity(clusters[i][j], centers[i], tanimoto_dict), 2)
            if j> max_length:
                max_length = j
    distance_matrix = distance_matrix[:,:max_length+1]
    obj_func = np.sum(distance_matrix)
    return obj_func
#                                                                             #
###############################################################################


###############################################################################
#calculates new clusterings from a set of centers                             #
#calculates new centers/means from a new set of seqs                      #
#                                                                             #
def k_means_clusters(seqs, clusters, centers, tanimoto_dict):
    distance_matrix = np.zeros([len(seqs), len(centers)])
    new_seqs = [[] for i in range(len(centers))]
    for i in range(len(seqs)):
        for j in range(len(centers)):
            distance_matrix[i,j] = compute_similarity(seqs[i], centers[j], tanimoto_dict)
    max_indices= np.argmin(distance_matrix, 1)
    for k in range(len(max_indices)):
        new_seqs[max_indices[k]].append(seqs[k])
    return new_seqs

def k_means_centers(seqs, centers, tanimoto_dict):
    
    #find distance from all points within a cluster to each other
    #or find distance from 
    new_centers = []
    for i in range(len(seqs)):
        distances = np.zeros([len(seqs[i]), 1])
        for j in range(len(seqs[i])):
            for k in range(len(seqs[i])):
                distances[j] = distances[j]+compute_similarity(seqs[i][j], seqs[i][k], tanimoto_dict)
        min_indices = np.argmin(distances)
        new_centers.append(seqs[i][min_indices])
    return new_centers

#                                                                             #
#                                                                             #
###############################################################################





###############################################################################
#                                                                             #
# Write the clustered ActiveSite instances out to a file.                     #
#                                                                             #
# Input: a filename and a clustering of ActiveSite instances                  #
# Output: none                                                                #

def write_clustering(filename, seqs):

    out = open(filename, 'w')

    for i in range(len(seqs)):
        out.write("\nCluster %d\n--------------\n" % i)
        for j in range(len(seqs[i])):
            out.write("%s\n" % seqs[i][j])

    out.close()

###############################################################################



###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
def main():
    np.set_printoptions(threshold=1000, linewidth=1000, precision = 5, suppress = False)
    directory = "C:\Users\Grace\Documents\GuanqingOuGoogleDrive\Backups\Berkeley\Classes\BMI203\bmi203-final-project\bmi203-final-project/"
    file = 'negatives.txt'
    read_sequences(directory+file)
    #clustering = cluster_by_partitioning(sequences)
    #write_clustering(targetfile, clustering)
    


###############################################################################

if __name__ == '__main__':
    main()