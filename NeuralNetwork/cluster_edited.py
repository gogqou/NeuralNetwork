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
#!C:/Python22/python.exe


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
        seq.nmers(3)
        sequenceList.append(seq)

    return sequenceList

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

def compute_similarity(seqA, seqB):
    similarity = distance.cityblock(seqA.vector_rep, seqB.vector_rep)
    return similarity

#                                                                             #
#                                                                             #
###############################################################################


########################################################################################################################################
#                                                                             #
# Cluster a given set of ActiveSite instances using a partitioning method.    #
#                                                                             #
# Input: a list of ActiveSite instances                                       #
# Output: a clustering of ActiveSite instances                                #
#         (this is really a list of clusters, each of which is list of        #
#         ActiveSite instances)                                               #

def cluster_by_partitioning(sequences):
    
    # randomly pick k centers
    #iterate through k and keep the lowest objective function clusters
    best_obj_func = 100000000
    #go through different values for number of clusters to find the one that gives the lowest obj function
    for k in range(20,len(sequences)/5, 150):
        #do a few tries at randomly generating centers
        for repeat in range(1):
            centers_indices = random.sample(xrange(0, len(sequences)), k)
            centers = []
            for i in range(len(centers_indices)):
                centers.append(sequences[centers_indices[i]])
            #centers just gives you the indices, not the actual instances
            #do k-means_clusters--this function takes the centers and clusters, calculates new clusters
            # k_means_centers takes new clusterings and calculates new centers
            current_clusters= k_means_clusters(sequences, sequences, centers)
            previous_epsilon = obj_function( current_clusters, centers)
            L = 0
            delta = previous_epsilon
            while delta>10 and L<600:
                current_clusters= k_means_clusters(sequences, current_clusters, centers)
                current_obj_func = obj_function(current_clusters, centers)
                centers = k_means_centers(current_clusters, centers)
                epsilon = current_obj_func-obj_function(current_clusters, centers)
                delta = abs(epsilon - previous_epsilon)
                previous_epsilon = epsilon
                print 'clustering iteration', L, 'k = ', k
                L = L+1
            if current_obj_func <best_obj_func:
                best_clusters = current_clusters
                best_centers= centers
                best_obj_func = current_obj_func
                best_k = k
            print current_obj_func, k
    #print best_k
    #print best_obj_func
    return best_clusters, centers
#                                                                             #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
#                                                                             #
def obj_function(clusters, centers):
    
    distance_matrix = np.zeros([len(centers),2000])
    max_length = 0
    for i in range(len(centers)):
        for j in range(len(clusters[i])):
            distance_matrix[i,j] = math.pow(compute_similarity(clusters[i][j], centers[i]), 2)
            if j> max_length:
                max_length = j
    distance_matrix = distance_matrix[:,:max_length+1]
    obj_func = np.sum(distance_matrix)
    return obj_func
#                                                                             #
###############################################################################


###############################################################################
#calculates new clusterings from a set of centers                             #
#calculates new centers/means from a new set of clusters                      #
#                                                                             #
def k_means_clusters(active_sites, clusters, centers):
    distance_matrix = np.zeros([len(active_sites), len(centers)])
    new_clusters = [[] for i in range(len(centers))]
    for i in range(len(active_sites)):
        for j in range(len(centers)):
            distance_matrix[i,j] = compute_similarity(active_sites[i], centers[j])
    max_indices= np.argmin(distance_matrix, 1)
    for k in range(len(max_indices)):
        new_clusters[max_indices[k]].append(active_sites[k])
    return new_clusters

def k_means_centers(clusters, centers):
    

    #find distance from all points within a cluster to each other
    #or find distance from 
    new_centers = []
    for i in range(len(clusters)):
        distances = np.zeros([len(clusters[i]), 1])
        for j in range(len(clusters[i])):
            for k in range(len(clusters[i])):
                distances[j] = distances[j]+compute_similarity(clusters[i][j], clusters[i][k])
        min_indices = np.argmin(distances)
        new_centers.append(clusters[i][min_indices])
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

def write_clustering(filename, clusters):

    out = open(filename, 'w')

    for i in range(len(clusters)):
        out.write("\nCluster %d\n--------------\n" % i)
        for j in range(len(clusters[i])):
            out.write("%s\n" % clusters[i][j].seq)

    out.close()

###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
def negative_centers():
    np.set_printoptions(threshold=1000, linewidth=1000, precision = 5, suppress = False)
    directory = "/home/gogqou/Documents/Classes/bmi203-final-project/"
    file = 'sample_nseqs.txt'
    targetfile = directory+'clustered_negatives.txt'
    sequenceList = read_sequences(directory+file)
    clustering, centers = cluster_by_partitioning(sequenceList)
    write_clustering(targetfile, clustering)
    print centers

    return centers
###############################################################################
