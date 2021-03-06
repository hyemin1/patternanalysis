import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bisect as bin
import numpy as np
import random as random
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




def parzen_reconstruct(input, windowSize):
    # reconstructs a sampled PDF using the parzen window estimator
    # This method is called Kernel Density Estimation (KDE) and helps us to estimate a PDF from a finite dataset.
    #           Parzen Window Estimator is a "non-parametric" way, i.e. that we do not assume a fixed underlying distribution such as a Gaussian.
    # Input is a 2D array containing values 0 (no sample) or 255 (sample)
    #       - This how the input data looks like: input[768][1024] with values 0 or 255
    #       - We just consider this input array as sample of a 2D PDF and want to reconstruct the PDF
    # Output is the same 2D array with the same dimensions like input with and intensity values 0..255 as reconstructed by the Parzen Window Estimator that comprise PDF values.


    height = len(input) # 768
    width = len(input[0]) # 1024

    print ("height: " + str(height))
    print ("width: " + str(width))



    D = 2 # 2-dimenstional
    h = windowSize
    V = pow(h, D)  # Volume of the hypercube V=h^D, in this case of 1D: V=h

    print("hypercube volume: " + str(V))
    print("windowSize: " + str(windowSize))

    totalSamples = 0
    for x in range(0, height):
        for y in range(0, width):
            if (input[x][y] == 255): ## brightness value is either 0 or 255
                totalSamples += 1

    print ("total number of samples: " + str(totalSamples))

    # the while loop below implements the following formula: p(x_i) = (1/N)* sum_{i=1..N} ((1/V) * Psi(x_i)*((x - x_i)/h))
    # with
    #   x: center of the hypercube
    #   x_i: position of one sample

    # Psi is just a function that is 1 if x_i is within the hypercube (i.e. all components of the vector are included); zero if outside.
    # We need to account for the problem that a dimension ("D: width, height) is not divisible by the window size. This is why we might use another window size for this exception.

    # we use this formula to calculate the pdf value for a given hyperbox: p(x) = k / (n*V) with k as the number of samples within a hyperbox. V the volume of the hyperbox and n the total number of samples.


    boxesX = height / windowSize
    if float(int(boxesX)) != boxesX: ## in case image dimension is not divisible by window size, we need to account for the small rest (later, we assign an alternative, smaller window size for this case)
        boxesX = int(boxesX) + 1

    boxesY = width / windowSize
    if float(int(boxesY)) != boxesY: ## in case image dimension is not divisible by window size, we need to account for the small rest (later, we assign an alternative, smaller window size for this case)
        boxesY = int(boxesY) + 1



    for bX in range (0, int(boxesX)):
        xMin = bX*windowSize
        xMax = min (xMin + windowSize - 1, height-1)
        for bY in range(0, int(boxesY)):
            yMin = bY * windowSize
            yMax = min(yMin + windowSize - 1, width - 1)
            currentBoxVolume = (xMax-xMin + 1) * (yMax-yMin + 1) # usually windowSize*windowSize, but it could be different at the edges

            centerX = (xMin + xMax)/ 2
            centerY = (yMin + yMax) / 2

            samplesWithinHyperbox = 0
            # now we check everything inside the box
            for x in range (xMin, xMax+1):
                for y in range(yMin, yMax + 1):
                    if (input[x][y] == 255):
                        samplesWithinHyperbox += 1

             # p(x) = k / (n * V)
            if (totalSamples * currentBoxVolume) > 0: # just to be safe....
                estimation = float(samplesWithinHyperbox) / (totalSamples * currentBoxVolume) # should be 0..1
            else:
                estimation = 0.0

            #print ("box: " + str(centerX) + "," + str(centerY) + " volume=" + str(currentBoxVolume) + " samples:"+str(samplesWithinHyperbox))

            # generate output: intensity values 0..255
            for x in range(xMin, xMax+1):
                for y in range(yMin, yMax+1):
                    input[x][y] = 255. * estimation
                    # We're directly replacing the input values here. That's bad practice but works because we don't need to access them again with the Parzen Window Estimator.


    return input




sampleSizes = (10000, 50000, 100000, 200000, 400000)
windowSizes = (5, 10, 20)
for s in range(0, len(sampleSizes)):
    sampledImage0 = raccoon_sample(sampleSizes[s])
    # Please mind: this image looks similar to a grayscale image. But actually it is just white dots on black background.
    # The higher a grayscale value of the original racoon image was, the more probable it is to become a white dot in this image.
    # This 2D array is what we take as samples to reconstruct a PDF with the help of the Parzen Window Estimator

    myplot = plt.imshow(sampledImage0)
    # plotImage (sampledImage)
    myplot.figure.savefig("samples=" + str(sampleSizes[s]) + "_raccoon-input.png")

    for w in range(0, len(windowSizes)):
        sampledImage = sampledImage0.copy()
        windowSize = windowSizes[w]
        reconstructedImage = parzen_reconstruct(sampledImage, windowSize)
        myplot = plt.imshow(reconstructedImage)
        # plotImage (reconstructedImage)
        myplot.figure.savefig("samples=" + str(sampleSizes[s]) + "_raccoon-reconstructed_h=" + str(windowSize) + ".png")
