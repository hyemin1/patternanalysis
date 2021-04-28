import numpy as np
import matplotlib.pyplot  as plt

#mean
mean = 1
#standard deviation
deviation = 0.2
#number of bins of the histogram
bin_num=30
#number of samples
sample_num=100
#generate random gaussian distributed samples
data = np.random.normal(mean,deviation,sample_num)

#plot the histogram with the data set and number of bins
plt.figure()
x,bin_data,others=plt.hist(data,bin_num)
#display histogram
plt.show()

#gaussian equation: (1/(deviation*root(2pi)))*exp(-(x-mean)^2/(2*deviation^2))
#plot gaussian curve
plt.plot(1/(deviation*np.sqrt(2*np.pi))*np.exp(-((bin_data-mean)**2)/(2*deviation**2)),color='r')
plt.show()
