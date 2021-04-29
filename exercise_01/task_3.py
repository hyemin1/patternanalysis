import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bisect as bin
import numpy as np
from sklearn import preprocessing as pre


####################
"""
This is an addition to task 2:
- 1. We generate  10.000, 50.000, 100.000, 200.000, and 400.000 samples from the racoon image and do not access the original racoon image anymore.
- 2. Then reconstruct the image with the Parzen Window Estimator the sampling the densities  for each of the five previously generated data sets.
- 3. Plot the output comprising of the density we got from the Parzen window estimator.
- 4. Play with window size h and answer question: What do you observe when you vary the window size of the Parzen estimator? => TODO

This is how the Parzen estimator works:
whole density: p(x) = 1/N sum(i=1..N) [(1/h^D)*k* ((x - x_i)/h)]
    -D: dimension
    -h: window size
    
"""
####################


def racoon_sample (sample_num):
    # generates sample_num from the racoon image
    
    ### this comes from task 2:
    raccoon = scipy.misc.face(gray=True)
    # gray coloring
    plt.gray()
    # apply gaussian filter
    raccoon = gaussian_filter(raccoon, sigma=3)
    # image display-- original raccon image
    plt.imshow(raccoon)
    # get rid of the axis
    plt.axis('off')
    plt.show()

    # convert 2D array to 1D
    raccoon = raccoon.flatten()
    # compute CDF
    raccoon_cdf = raccoon.cumsum()
    #normalization skipped


    # compute sampled array
    # output array (1-D)
    output= [0]*len(raccoon_cdf)


    #generate random, uniform samples
    u_data = np.random.uniform(0, max(raccoon_cdf), sample_num)
    # sort u_data in non-decreasing order
    u_data = sorted(u_data)

    #u_crr for indexing u_data
    u_crr=0

    #loop through CDF
    for x in range(len(raccoon_cdf)):
        #if we checked all values in u_data, end loop
        if u_crr==len(u_data):
            break
        #if current CDF value is equal or bigger than current value of u_data
        if(raccoon_cdf[x]>=u_data[u_crr]):
            output[x]=raccoon_cdf[x]
            u_crr+=1
            
    return output


def parzen_reconstruct (input):
    # reconstructs a sampled image using the parzen window estimator
    h = 123
    D = 2
    
    """
    todo...
    whole density: p(x) = 1/N sum(i=1..N) [(1/h^D)*k* ((x - x_i)/h)]    
    """
    
   return 0 # todo...




# reconstruct images from 10k, 50k, 100k, 200k and 400k samples using the Parzen window estimator
recon10k = parzen_reconstruct(racoon_sample (10000))
recon50k = parzen_reconstruct(racoon_sample (50000))
recon100k = parzen_reconstruct(racoon_sample (100000))
recon200k = parzen_reconstruct(racoon_sample (200000))
recon400k = parzen_reconstruct(racoon_sample (400000))



#reshape array to 2D
output10k=np.reshape(recon10k,(768,1024))
output50k=np.reshape(recon50k,(768,1024))
output100k=np.reshape(recon100k,(768,1024))
output200k=np.reshape(recon200k,(768,1024))
output400k=np.reshape(recon400k,(768,1024))


#display
plt.figure()
plt.imshow(output400k) # multi plot....
plt.show()
