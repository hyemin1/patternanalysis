# patternanalysis
Pattern Analysis, 2021 Summer Semester, FAU



exercise_01

task 1- warm up
Display a histogram with normally distributed data samples and a gaussian curve

task2- practice sampling algorithm
Sample a racoon image using CDF

task3-Parzen Window
Addition to task 2, reconstruct sampled image using parzen window. Check the differences caused by the window size.
-> As the window size grows, we can find more correctly described raccoon image.(reduced overfitting) But if the window size is too big, underfitting can occur. 

task4-optimal Parzen Window Size
Find optimal Parzen Window Size that gives high fit quality in general.
-> Window size should be at least 20 in our case-> stable fit quality for various number of samples
