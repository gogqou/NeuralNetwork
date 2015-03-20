'''
Created on Mar 19, 2015

@author: gogqou
'''
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
# A simple class for an atom                                                  #

class Atom:

    def __init__(self, type):
        self.type = type
        self.coords = [0.0, 0.0, 0.0]

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.type

###############################################################################



###############################################################################
#                                                                             #
# A simple class for an amino acid residue                                    #

class Residue:

    def __init__(self, type, number):
        self.type = type
        self.number = number
        self.atoms = []
        self.sum_coords=[0.0, 0.0, 0.0]
        self.avg_coords=[0.0, 0.0, 0.0]
        self.max_atom=[]
        self.min_atom = []

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.type + " " + self.number

###############################################################################



###############################################################################
#                                                                             #
# A simple class for an active site                                           #

class ActiveSite:

    def __init__(self, name):
        self.name = name
        self.residues = []
        self.multimer = False
        self.nmer = 1
        self.sum_coords = [0.0, 0.0, 0.0]
        self.center=[0.0, 0.0, 0.0] #Euclidean center of active site--average pos of residues 
        self.monomersize = 0 #number of residues consisting monomer repeat
        self.avg_res_dist = 0 #average residue distance from active site center
        self.stdev_res_dist = 0 #st deviation of residue distance from site center
        self.farthest_res = 0 #distance of residue farthest from center
        self.nearest_res = 0 #distance of residue closest to center
        self.unique_res = set() #list of unique residues making up the active site
        self.metric =0

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.name

###############################################################################



###############################################################################
#                                                                             #
# Read in all of the active sites from the given directory.                   #
#                                                                             #
# Input: directory   
# Processing: finding average coordinates for each residue and active site    #
# Output: list of ActiveSite instances                                        #

def read_active_sites(dir):
    files = glob.glob(dir + '/*.pdb')
    active_sites = []
    for file in files:
        if os.name == 'nt':
            name = splitfields(file, '\\')[-1][:-4]
        else:
            name = splitfields(file, '/')[-1][:-4]
        active_site = ActiveSite(name)
        
        l = open(file, "r").readlines()
        r_num = 0
        for i in range(len(l)):
            t = split(l[i])
            if t[0] != 'TER':
    
                # read in an atom
                atom_type = l[i][13:17]
                x_coord = float(l[i][30:38])
                y_coord = float(l[i][38:46])
                z_coord = float(l[i][46:54])
                atom = Atom(atom_type)
                atom.coords = [x_coord, y_coord, z_coord]
                  
                residue_type = l[i][17:20]
                residue_number = l[i][23:26]
            
                # make a new residue if needed
                if residue_number != r_num:
                    residue = Residue(residue_type, residue_number)
                    r_num = residue_number
    
                # add the atom to the residue
                residue.atoms.append(atom)
                residue.sum_coords = np.add(residue.sum_coords, atom.coords)
          
            else:  # I've reached a TER card
                active_site.residues.append(residue)
                temp_atoms = residue.atoms 
                residue.avg_coords = residue.sum_coords/len(temp_atoms)
                active_site.sum_coords =np.add(residue.avg_coords, active_site.sum_coords)
        temp_residues = active_site.residues 
        active_site.center = active_site.sum_coords/len(temp_residues)   
        active_sites.append(active_site)
        
    
    print "Read in %d active sites" % len(active_sites)
    residue_dist_center(active_sites)
    print 'Calculated center and residue distances'
    return active_sites
#                                                                             #
#                                                                             #
###############################################################################


#####################################################################################
#check if there are residue num repeats--tells us it's a multimer                   #
#this can start as a first pass at segregating active sites                         #
#returns "check" == True if it is a multimer, and n for how many repeats it contains#
def check_multimer(site):
    n = 1
    check = False
    residues = site.residues
    monomer_num = len(residues)
    for i in range(1,len(residues)):
        if residues[0].number==residues[i].number:
            monomer_num = min(monomer_num, i) 
            #if this is a multimer, count how many residues consist the monomer
            #right now this gives the index of the first repeat, which gives
            #the right answer if we're looking for size of monomer, but remember 
            #to subtract by one if you want the actual index of the last non repeat
            n = n+1 #every time we hit a repeat, we increment the count for the 
            #number of monomers
            check = True  
            #if we hit a repeat, we change the check to True, meaning this is a multimer
    return check, n, monomer_num

####################################################################################
def nmers(active_sites):
    
    nmers = [1]
    labeled_clusters = []
    for j in range(len(active_sites)):
        [check, n, monomer_num]=check_multimer(active_sites[j])
        active_sites[j].monomersize= monomer_num #assigns the monomer size from count performed in check_multimer
        if check is True:
            active_sites[j].multimer = True
            active_sites[j].nmer = n
            if n in nmers:
                continue
            else:
                nmers.append(n)
    
    nmers = sorted(nmers)
    clusters = [[] for i in range(len(nmers))]
    labeled_clusters = zip(nmers, clusters)
    for m in range(len(nmers)):
        for j in range(len(active_sites)):
            if active_sites[j].nmer == nmers[m]:
                clusters[m].append(active_sites[j])    
    for k in range(len(clusters)):
        print labeled_clusters[k]
    
    return labeled_clusters, active_sites



###############################################################################
# calculates the distance between two residues primary carbons                #
#                                                                             #
def compute_Carbon_dist(res1, res2):
    res1C = res1.atoms[2]
    res2C = res2.atoms[2]
    dist = distance.euclidean(res1C.coords, res2C.coords)
    return dist
###############################################################################
# calculates the distance between every residue and the site center           #
# finds shortest and longest distance from center to residue                  #
# since we're looping through anyway, count unique residues                   #
def residue_dist_center(active_sites):
    max_avg_dist = 0
    max_farthest_res = 0
    max_nearest_res = 0
    for i in range(len(active_sites)):
        residues = active_sites[i].residues
        unique_residues = set()
        distance_vector = np.zeros([len(residues), 1])
        for j in range(len(residues)):
            if residues[j].type not in unique_residues:
                unique_residues.add(residues[j].type)
                
            #choose this way of calculating distance if want to use avg coords
            # of all the atoms in the residue
            #distance_vector[j] = distance.euclidean(active_sites[i].center, residues[j].avg_coords)
            
            #choose this way of calculating distance if want to use the central 
            #carbon as the representative atom of the residue
            distance_vector[j] = distance.euclidean(active_sites[i].center, residues[j].atoms[2].coords)
        active_sites[i].unique_res = unique_residues
        active_sites[i].avg_res_dist = np.sum(distance_vector)/len(residues)
        active_sites[i].stdev_res_dist = np.std(distance_vector)
        active_sites[i].farthest_res = np.amax(distance_vector)
        active_sites[i].nearest_res = np.amin(distance_vector)
        if active_sites[i].avg_res_dist>max_avg_dist:
            max_avg_dist = active_sites[i].avg_res_dist
        if active_sites[i].farthest_res>max_farthest_res:
            max_farthest_res=active_sites[i].farthest_res
        if active_sites[i].nearest_res> max_nearest_res:
            max_nearest_res=active_sites[i].nearest_res
    for i in range(len(active_sites)):
        active_sites[i].avg_res_dist = active_sites[i].avg_res_dist/max_avg_dist
        active_sites[i].stdev_res_dist = active_sites[i].stdev_res_dist/max_avg_dist
        active_sites[i].farthest_res = active_sites[i].farthest_res/max_farthest_res
        active_sites[i].nearest_res = active_sites[i].nearest_res/max_nearest_res
    return active_sites

###################################################################################
#
#calculate the tanimoto coefficient based on comparison of AAs in two active sites#
#                                                                                 #
def tanimoto(setA, setB):
    unionAB = list(setA | setB)
    intersectAB = setA.intersection(setB)
    tanimoto_coeff=len(intersectAB)/float(len(unionAB))    
    return tanimoto_coeff

def tanimoto_sites(active_sites):
    tanimoto_dict = {}
    for i in range(len(active_sites)):
        for j in range(len(active_sites)):
            tanimoto_coeff = tanimoto(active_sites[i].unique_res,active_sites[j].unique_res)
            if tanimoto_coeff>0:
                tanimoto_dict[(active_sites[i].name, active_sites[j].name)] =1-log(tanimoto_coeff, 2)
                tanimoto_dict[(active_sites[j].name, active_sites[i].name)] =1-log(tanimoto_coeff, 2)
            else:
                tanimoto_dict[(active_sites[i].name, active_sites[j].name)] =1
                tanimoto_dict[(active_sites[j].name, active_sites[i].name)] =1
    return tanimoto_dict

###############################################################################
###############################################################################
# Iterates through and compiles the similarity metric values                  #
#                                                                             #
#                                                                             #
def similarity_metric(active_sites):
    for i in range(len(active_sites)):
        active_sites[i].metric = np.array([active_sites[i].avg_res_dist,active_sites[i].stdev_res_dist, active_sites[i].farthest_res, active_sites[i].nearest_res, active_sites[i].nmer])    
    return active_sites

###############################################################################

###############################################################################
#                                                                             #
# Compute the similarity between two given ActiveSite instances.              #
#                                                                             #
# Input: two ActiveSite instances                                             #
# Output: the similarity between them (a floating point number)               #
#                                                                             #

def compute_similarity(site_A, site_B, tanimoto_dict):
    similarity = tanimoto_dict[(site_A.name, site_B.name)]+distance.euclidean(site_A.metric, site_B.metric)
    #similarity = distance.euclidean(site_A.metric, site_B.metric)
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

def cluster_by_partitioning(active_sites):
    

    # Part of the distance metric will be the multimer-state of the enzyme site
    #so first calculate that and save it as a feature 
    [orig_clusters, active_sites] = nmers(active_sites)
    #get the tanimoto dictionary for comparisons of all active sites to all other active sites
    tanimoto_dict = tanimoto_sites(active_sites)
    active_sites = similarity_metric(active_sites)
    # randomly pick k centers
    #iterate through k and keep the lowest objective function clusters
    best_obj_func = 10000
    #go through different values for number of clusters to find the one that gives the lowest obj function
    for k in range(1,len(active_sites)/4, 3):
        #do a few tries at randomly generating centers
        for repeat in range(4):
            centers_indices = random.sample(xrange(0, len(active_sites)), k)
            centers = []
            for i in range(len(centers_indices)):
                centers.append(active_sites[centers_indices[i]])
            #centers just gives you the indices, not the actual instances
            #do k-means_clusters--this function takes the centers and clusters, calculates new clusters
            # k_means_centers takes new clusterings and calculates new centers
            current_clusters= k_means_clusters(active_sites, orig_clusters, centers, tanimoto_dict)
            previous_epsilon = obj_function( current_clusters, centers, tanimoto_dict)
            L = 0
            delta = previous_epsilon
            while delta>10 and L<600:
                current_clusters= k_means_clusters(active_sites, current_clusters, centers, tanimoto_dict)
                current_obj_func = obj_function(current_clusters, centers,tanimoto_dict)
                centers = k_means_centers(current_clusters, centers, tanimoto_dict)
                epsilon = current_obj_func-obj_function(current_clusters, centers,tanimoto_dict)
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
#calculates new centers/means from a new set of clusters                      #
#                                                                             #
def k_means_clusters(active_sites, clusters, centers, tanimoto_dict):
    distance_matrix = np.zeros([len(active_sites), len(centers)])
    new_clusters = [[] for i in range(len(centers))]
    for i in range(len(active_sites)):
        for j in range(len(centers)):
            distance_matrix[i,j] = compute_similarity(active_sites[i], centers[j], tanimoto_dict)
    max_indices= np.argmin(distance_matrix, 1)
    for k in range(len(max_indices)):
        new_clusters[max_indices[k]].append(active_sites[k])
    return new_clusters

def k_means_centers(clusters, centers, tanimoto_dict):
    
    
    # WORK IN PROGRESS NEED TO CHANGE CALCULATIONS AFTER PUTTING IN ALL 
    #THE PARTS OF THE SIMILARITY METRIC
    #find distance from all points within a cluster to each other
    #or find distance from 
    new_centers = []
    for i in range(len(clusters)):
        distances = np.zeros([len(clusters[i]), 1])
        for j in range(len(clusters[i])):
            for k in range(len(clusters[i])):
                distances[j] = distances[j]+compute_similarity(clusters[i][j], clusters[i][k], tanimoto_dict)
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
            out.write("%s\n" % clusters[i][j])

    out.close()

###############################################################################



###############################################################################
#                                                                             #
# Write a series of clusterings of ActiveSite instances out to a file.        #
#                                                                             #
# Input: a filename and a list of clusterings of ActiveSite instances         #
# Output: none                                                                #

def write_mult_clusterings(filename, clusterings):

    out = open(filename, 'w')

    for i in range(len(clusterings)):
        clusters = clusterings[i]
        for j in range(len(clusters)):
            out.write("\nClustering %d\n---------------***--------------------------\n" % i)
            for k in range(len(clusters[j])):
                out.write("\nCluster %d\n------------\n" % k)
                for m in range(len(clusters[j][k])):
                    
                    out.write("%s\n" % clusters[j][k][m])

    out.close()

###############################################################################





###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
def main():
    np.set_printoptions(threshold=1000, linewidth=1000, precision = 5, suppress = False)
    
    # Some quick stuff to make sure the program is called correctly
    if len(sys.argv) < 4:
        print "Usage: cluster.py [-P| -H] <pdb directory> <output file>"
        sys.exit(0)
    
###############################################################################
#                                                                             #
# Top Level                                                                   #

    active_sites = read_active_sites(sys.argv[2])
    
    # Choose clustering algorithm
    if sys.argv[1][0:2] == '-P':
        print "Clustering using Partitioning method"
        clustering = cluster_by_partitioning(active_sites)
        write_clustering(sys.argv[3], clustering)
    
    if sys.argv[1][0:2] == '-H':
        print "Clustering using hierarchical method"
        clusterings = cluster_hierarchically(active_sites)
        write_mult_clusterings(sys.argv[3], clusterings)


###############################################################################
    
    return 1
if __name__ == '__main__':
    main()