#Import the required packages

from __future__ import print_function

#numpy
import numpy as np

#os
import os

#glob
import glob


#data path
superfolder = "feature_vectors"
file_smile = "smile_vec.npy"
file_not_smile = "not_smile_vec.npy"

#Load the data
smile_vec = np.load(os.path.join(superfolder, file_smile))
not_smile_vec = np.load(os.path.join(superfolder, file_not_smile))

print("The shape of the smile-feature-vector-matrix is:")
print(smile_vec.shape)
print("The shape of the nonsmile-feature-vector-matrix is:")
print(not_smile_vec.shape)

#function to calculate the mean of all given feature vectors (vectors)
def get_mean(vectors):
    mean = vectors[0,:,:]
    for i in range(1, vectors.shape[0]):
        mean = mean + vectors[i, :, :]
    mean = np.divide(mean, smile_vec.shape[0])
    return mean


#Calculate the mean coords
mean_smile_vec = get_mean(smile_vec)
mean_not_smile_vec = get_mean(not_smile_vec)

#Translationvector
translation_to_smile = mean_smile_vec - mean_not_smile_vec
np.save('translation_to_smile_vec.npy', translation_to_smile)
print("Length of the translationvector")
print(np.linalg.norm(translation_to_smile))

#empirical variances
variance_smile_vec = np.var(smile_vec, axis=0, ddof=1)
variance_not_smile_vec = np.var(not_smile_vec, axis=0, ddof=1)

#empirical std
std_smile_vec = np.sqrt(variance_smile_vec)
std_not_smile_vec = np.sqrt(variance_not_smile_vec)

#maximum variance and std
max_variance_smile_vec = np.amax(variance_smile_vec)
max_variance_not_smile_vec = np.amax(variance_not_smile_vec)
max_std_smile_vec = np.sqrt(max_variance_smile_vec)
max_std_not_smile_vec = np.sqrt(max_variance_not_smile_vec)

#Norm of variances
norm_variance_smile_vec = np.linalg.norm(variance_smile_vec)
norm_variance_not_smile_vec = np.linalg.norm(variance_not_smile_vec)
#print(norm_variance_smile_vec)
#print(norm_variance_not_smile_vec)

#Norm of empirical std
norm_std_smile_vec = np.linalg.norm(std_smile_vec)
norm_std_not_smile_vec = np.linalg.norm(std_not_smile_vec)
print("Norm of empirical std (smiling):")
print(norm_std_smile_vec)
print("Norm of empirical std (nonsmiling):")
print(norm_std_not_smile_vec)

#Proof, how many points are in their opposite regime

#Nonsmile (68% confident)
not_smile_counter = 0
for not_smile in not_smile_vec:
    diff_vec = (not_smile - mean_smile_vec)
    if( np.linalg.norm(diff_vec) <= (norm_std_smile_vec) ):
        not_smile_counter += 1

print("nonsmiling-feature vectors in smiling regime:")        
print(not_smile_counter)

#Smile (68% confident)
smile_counter = 0
for smile in smile_vec:
    diff_vec = (smile - mean_not_smile_vec)
    if( np.linalg.norm(diff_vec) <= (norm_std_not_smile_vec) ):
        smile_counter += 1

print("smiling-feature vectors in nonsmiling regime:")        
print(smile_counter)

print('__________')
not_smile_perc = float(not_smile_counter)/float(not_smile_vec.shape[0])
smile_perc = float(smile_counter)/float(smile_vec.shape[0])
print("percentage of nonsmiling-feature vectors in smiling regime:") 
print(not_smile_perc)
print("percentage of smiling-feature vectors in nonsmiling regime:")
print(smile_perc)

#Proof, how many points are in their own regimes

#Nonsmile (68% confident)
not_smile_counter = 0
for not_smile in not_smile_vec:
    diff_vec = (not_smile - mean_not_smile_vec)
    #print(np.linalg.norm(diff_vec))
    if( np.linalg.norm(diff_vec) <= (norm_std_smile_vec) ):
        not_smile_counter += 1

print("nonsmiling-feature vectors in nonsmiling regime:")         
print(not_smile_counter)

#Smile (68% confident)
smile_counter = 0
for smile in smile_vec:
    diff_vec = (smile - mean_smile_vec)
    if( np.linalg.norm(diff_vec) <= (norm_std_not_smile_vec) ):
        smile_counter += 1

print("smiling-feature vectors in smiling regime:")         
print(smile_counter)

print('__________')
not_smile_perc = float(not_smile_counter)/float(not_smile_vec.shape[0])
smile_perc = float(smile_counter)/float(smile_vec.shape[0])
print("percentage of nonsmiling-feature vectors in nonsmiling regime:") 
print(not_smile_perc)
print("percentage of smiling-feature vectors in smiling regime:") 
print(smile_perc)

