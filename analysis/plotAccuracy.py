"""
File description : Plots accuracy evolution of training set vs. testing set
                   with different configurations of either embedding, or cells
                   configs.
"""
import matplotlib.pyplot as plt
import numpy as np


def plotAccuracy(allAccuracies) :
    """ Plots accuracies against epochs
    Input : allAccuracies : list of lists
    Outputs : Plot file

    indice 1 --> Basic RNN tanh
    indice 2 --> LSTM + two embedded layers
    indice 3 --> bi-GRU layers and 2 embedded layers

    Max epoch : ?
    """


    
