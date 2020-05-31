"""
File description : convergence analysis ----------------------------------------
                   For each input data folder, builds the mean accuracy (test)
                   array, then computes the associated std per epoch

"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


FILEPATH = "allConvergence/"
maxData = 1
amountExperiments = 10

allData = []

for i in range(maxData) :
    string = FILEPATH + str(i) + '.csv'
    # read csv file as a list of lists
    with open(string) as csvfile:


y1 = []
y2 = []




plt.show()
