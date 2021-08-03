# patternanalysis
Pattern Analysis, 2021 Summer Semester, FAU



exercise_01: Random Sampling & Parzen Window

task 1- warm up
_Display a histogram with normally distributed data samples and a gaussian curve

task2- practice sampling algorithm
_Sample a racoon image using CDF

task3-Parzen Window
_Addition to task 2, reconstruct sampled image using parzen window. Check the differences caused by the window size.
-> As the window size grows, we can find more correctly described raccoon image.(reduced overfitting) But if the window size is too big, underfitting can occur. 

task4-optimal Parzen Window Size
_Find optimal Parzen Window Size that gives high fit quality in general.
-> Window size should be at least 20 in our case-> stable fit quality for various number of samples



exercise_02: Random Tree & Extra Tree Regression
Task 1:
_add unifrom distribution to background
_fit the regression model using random forest

Task 2:
_try to fit the model with extra tree regressor

-> Extra Tree Regressor gives better result than Random Tree Regressor
->Reason: Extra Tree Regressor contains split subtress randomly and doesn't contain bootstrapping, but Random Tree Regressor chooses the best subtree and does bootstrapping computation. 
->As Extra Tree Regressor does more random computation, the result has less variance than that of Random Tree Regressor.
-> The result also depends on hyper parameters like maximum depth of tree, number of estimators, etc.
