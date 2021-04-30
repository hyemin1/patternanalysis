import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bisect as bin
import numpy as np
from sklearn import preprocessing as pre


#########
""" 
task 4: use code from task 3 and find the optimal windows size h
  - use a fixed sample size and generate racoon samples like in task 3.
  - recontruct the image with the Parzen Window Estimator and different window sizes h
  - evaluate and quantify how well this worked for a certain window size h
  - try out different window sizes and find the optimal value
  - want to check: does h depend so sample size => possibly no, but we want to test it.
"""
#########



def evaluateReconstruction (reconstruction):
  # quantifies how well the image has been reconstructed for a certain window size
  # for this, we use cross-correlation:
  #     how cross-validation works
  #         -split samples S into
  #	            - a test set S_j of size N/k
  #	            - a training set T_j
  #         -build a tensity p(x)_Theta^j from T_j
  #	            -Theta: candidate kernel size (h?)
  #         - choose the best Theta with ML estimation across all folds with
  #	            -Theta = argmax_Theta sum [log p(x)_Theta^j] on all x € T_j across all folds j
  #         -make sure to split samples randomly between folds!
  
  return -1.23


standardSample = racoon_sample (100000)
bestWindowSize = -1
bestEvaluation = -1

for h in range(5 to 50):
    evaluation = evaluateReconstruction (parzen_reconstruct (standardSample, h))
    if (bestWindowSize < 0 or bestEvaluation > evaluation):
      bestWindowSize = h
      bestEvaluation = evaluation
      

 print ("optimal window size for parson estimator with racoon image: " + str(bestWindowSize))
    
