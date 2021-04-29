import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bisect as bin
import numpy as np
from sklearn import preprocessing as pre


####################
"""
This is an addition to task 2:
- 1. We have generate  10.000, 50.000, 100.000, 200.000, and 400.000 samples from the racoon image and do not access the original racoon image anymore.
- 2. Then reconstruct the image with the Parzen Window Estimator the sampling the densities  for each of the five previously generated data sets.
- 3. Plot the output comprising of the density we got from the Parzen window estimator.
"""
####################




# load the racoon image
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
# number of samples
sample_num = 500000

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

#reshape array to 2-D
output=np.reshape(output,(768,1024))

#display
plt.figure()
plt.imshow(output)

plt.show()
