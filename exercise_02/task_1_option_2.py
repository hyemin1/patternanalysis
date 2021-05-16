import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
##################
"""
exercise 2, task 1
1. foreground: load raccoon &do random sampling
2. background: create background image using uniform distribution
3. apply random forest regression
"""
##################

#load raccoon image
def load_raccoon():
    raccoon = scipy.misc.face(gray=True)
    # gray coloring
    plt.gray()
    # apply gaussian filter
    raccoon = gaussian_filter(raccoon, sigma=3)
    return raccoon

#create background array
def create_background(raccoon):
    background = np.random.uniform(0,len(raccoon),768*1024)
    background= np.reshape(background,(768,1024))
    return background

#do random forest regression
def randomregression(foreground,background):
    #create traingin & test sets
    trainX, testX, trainY, testY = train_test_split(background, foreground,train_size=0.6)
    #do random forest regression
    #set regressor
    reg = RandomForestRegressor(n_estimators=20, max_depth=10)
    #do training
    reg.fit(trainX, trainY)
    #get the predicted output
    after_tree_image = reg.predict(background)
    #display
    plotImage(after_tree_image,"Random Forest Regression")

#do random sampling
def raccoon_sample(sample_num,raccoon):
    # convert 2D raccoon array to 1D
    raccoon = raccoon.flatten()
    # compute CDF
    raccoon_cdf = raccoon.cumsum()

    # normalization skipped
    # compute sampled array
    # output array (1-D)
    output = [0] * len(raccoon_cdf)
    # number of samples

    # generate random, uniform samples
    u_data = np.random.uniform(0, max(raccoon_cdf), sample_num)
    # sort u_data in non-decreasing order
    u_data = sorted(u_data)

    # u_crr for indexing u_data
    u_crr = 0

    # loop through CDG
    for x in range(len(raccoon_cdf)):
        # if we checked all values in u_data, end loop
        if u_crr == len(u_data):
            break
        # if current CDF value is equal or bigger than current value of u_data
        if (raccoon_cdf[x] >= u_data[u_crr]):
            output[x] = raccoon_cdf[x]
            u_crr += 1

    # reshape array to 2-D
    output = np.reshape(output, (768, 1024))

    return output

#display
def plotImage (image,img_title):

    # display
    #plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.title(img_title)
    plt.show()

"""
main
"""
#foreground
#load raccoon image
foreground= load_raccoon()
#do random sampling
foreground=raccoon_sample(200000,foreground)
#display
plotImage(foreground,"foreground image")
#background
#create background array
background = create_background(foreground)
#do random sampling
background = raccoon_sample(200000,background)
#display
plotImage(background,"background image")
#do random forest regression & display
randomregression(foreground,background)
