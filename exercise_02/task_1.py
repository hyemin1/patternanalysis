import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bisect as bin
import numpy as np
import random as random
from sklearn import preprocessing as pre

####################
"""
exercise 2, task 1

"""


####################


def raccoon_sample(sample_num):
    # generates sample_num samples from a probability distribution obtained from brightness values of a racoon image turned into 1 1D distribution
    # note: output is 2D with 0 or 255 brightness value.

    ### this comes from task 2:
    raccoon = scipy.misc.face(gray=True)
    # gray coloring
    plt.gray()
    # apply gaussian filter
    raccoon = gaussian_filter(raccoon, sigma=3)

    myplot = plt.imshow(raccoon)
    # plotImage (reconstructedImage)
    myplot.figure.savefig("raccoon_gaussian3sigma_original.png")

    # print(len(raccoon)) # => 768
    # print(len(raccoon[0]))  # => 1024
    # print(min(raccoon[100]))
    # print(max(raccoon[100]))

    #### this is how it looks like: racoon[768][1024] with values 0..255

    output = np.zeros((768, 1024))
    for x in range(0,768):
        for y in range(0,1024):
            output[x][y] = -1

    #draw random samples: output[768][1024] is -1 if no sample is drawn an 0..255 if it is one of the grayscale values originating from the raccoon image
    s = 0
    while (s < sample_num):
        x = np.random.randint(0, 768-1)
        y = np.random.randint (0, 1024-1)
        if output[x][y] == -1: # pixel not used yet
            output[x][y] = 0
            if random.random() >= float(raccoon[x][y])/255.: # probability depends on brightness value
                output[x][y] = 255
                s+=1

    return output


def plotImage (image):
    # typical input: image[768][1024] wih integer values 0..255
    # pixel values of -1 are also okay and just displayed black


    # display
    plt.figure()
    plt.imshow(image)

    plt.show()


