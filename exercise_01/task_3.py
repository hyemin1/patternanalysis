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
- 2. Then reconstruct the image with the Parzen Window Estimator the sampling the densities for each of the five previously generated data sets.
- 3. Plot the output comprising of the density we got from the Parzen window estimator.
- 4. Play with window size h and answer question: What do you observe when you vary the window size of the Parzen estimator? => TODO
This is how the Parzen estimator works:
whole density: p(x) = 1/N sum(i=1..N) [(1/h^D)*k* ((x - x_i)/h)]
    -D: dimension
    -h: window size
    
"""
####################


def racoon_sample (sample_num):
    # generates sample_num samples from a probability distribution obtained from brightness values of a racoon image turned into 1 1D distribution
    # note: output is 1D
    
    ### this comes from task 2:
    raccoon = scipy.misc.face(gray=True)
    # gray coloring
    plt.gray()
    # apply gaussian filter
    raccoon = gaussian_filter(raccoon, sigma=3)

    """
    # image display-- original raccon image
    plt.imshow(raccoon)
    # get rid of the axis
    plt.axis('off')
    plt.show()
    """

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

    # output is an array containing 1D positions (they came from 2D->1D by generating a pdf from the racoon image) of where we sampled the PDF.
    return output


def parzen_reconstruct (input, windowSize):
    # reconstructs a sampled PDF using the parzen window estimator
    # This method is called Kernel Density Estimation (KDE) and helps us to estimate a PDF from a finite dataset.
    #           Parzen Window Estimator is a "non-parametric" way, i.e. that we do not assume a fixed underlying distribution such as a Gaussian.
    # Input is a 1D array containing grayscale values at positions that were sampled and zeros at positions that have not been sampled.
    #       - A 1D position can later be transformed into 2D again.
    #       - This is an example of how the input data looks like: [0, 0, 0, (...), 0, 0, 88148563, 0, 0, 0, 0, 0, 0, 88148812, 0, 88148123, 0, 0, 0, 0, 0]
    # Output is a 1D array of the same size like the input. Instead of the black spots that were represented by zero entries,
    # this function calculated how the data should actually look like based on the Parzen Window Estimator.
    #       - This array represents a PDF
    #       - It can be turned into a 2D array as their 1D coordinate number transforms into a pair of x-y-coordinates
    #       - I.e., the PDF of which we only had a limited number of samples has been reconstructed and can be turned into an image again.

    N = len (input) # N is the number of all available samples
    h = windowSize
    V = pow(h, 1) # Volume of the hypercube V=h^D, in this case of 1D: V=h


    # the while loop below implements the folowing formula: p(x_i) = (1/N)* sum_{i=1..N} ((1/V) * Psi(x_i)*((x - x_i)/h))
    # with
    #   x: center of the hypercube
    #   x_i: PDF value of one sample
    #   i=0..N-1 with N being the number of available samples
    #   V=h^D as the volume of the hypercube, in case of 1D: V=h
    #   Psi is just a function that is 1 if x_i is within the hypercube (i.e. all components of the vector are included); zero if outside.
    # so practically, for one hyperbox we can calculate p(x) = (1/(N*V)) * sum_{over all x_i in the hyperbox} (x - x_i) / h
    # then we assign this p(x) value to all x_i that are within the hyperbox
    # the only problem is that the last window might be smaller than h because there is no more data. That's why we might use another window size for this exception
    windowRight = windowSize - 1

    H = int(N / h)
    if N%h: H+=1 #  we need to account for the last window that might be a little smaller than intented windowSize

    output = input # we're just copying this to avoid operating on input[] while we're still reading it - should be fine but just to be safe.
    windowLeft = 0
    for b in range (0, H): # iterate over all boxes
        windowRight = windowLeft + h - 1
        if (windowRight >= N):
            windowRight = N - 1
        currentWindowSize = (windowRight - windowLeft) + 1 # in most cases, this is the original window size h; during the last iteration it migh tbe smaller
        boxCenter = (windowRight + windowLeft) / 2. # does not need to be an integer

        # now we calculate the box value and iterate over all samples within the hyperbox - this covers the Psi part of the equation
        boxValue = 0
        for i in range (windowLeft, windowRight + 1):
            boxValue += (boxCenter - i) / currentWindowSize ### PLEASE CHECK: do we need to multiply this with input[i]? Somehow we need to get in the grayscale values. Feels right but I can't find this in the formula.
            #boxValue += (boxCenter - i) / currentWindowSize

        # now we assign the box value to all the points within the current window
        for i in range (windowLeft, windowRight + 1):
            output[i] = boxValue*input[i]


        windowLeft += h #  for the next iteration



    return output



# check how the sampled data flooks like:
#testdata = racoon_sample (10000)
#print (testdata)
# => [0, 0, 0, (...), 0, 0, 88148563, 0, 0, 0, 0, 0, 0, 88148812, 0, 88148123, 0, 0, 0, 0, 0]
# this is a 1D array with mostly zeros and occasionally grayscale values; the more samples we drew, the more often non-zero entries occur.




# reconstruct images from 10k, 50k, 100k, 200k and 400k samples using the Parzen window estimator

windowSize = 100
recon10k = parzen_reconstruct(racoon_sample (10000), windowSize)
recon50k = parzen_reconstruct(racoon_sample (50000), windowSize)
recon100k = parzen_reconstruct(racoon_sample (100000), windowSize)
recon200k = parzen_reconstruct(racoon_sample (200000),windowSize)
recon400k = parzen_reconstruct(racoon_sample (400000), windowSize)



#reshape the reconstructed array to 2D
output10k=np.reshape(recon10k,(768,1024))
output50k=np.reshape(recon50k,(768,1024))
output100k=np.reshape(recon100k,(768,1024))
output200k=np.reshape(recon200k,(768,1024))
output400k=np.reshape(recon400k,(768,1024))



#display
plt.figure()
plt.imshow(output10k)
plt.show()

plt.imshow(output50k)
plt.show()

plt.imshow(output100k)
plt.show()

plt.imshow(output200k)
plt.show()

plt.imshow(output400k)
plt.show()

## Parzen Window Reconstruction does what it is supposed to do. Result looks bad because we use 1D and could be fixed by applying another way to go 2D-1D.







