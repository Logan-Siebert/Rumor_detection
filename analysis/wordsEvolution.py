"""
File description : Plotting the k-most important data evolution per timestamp.
                   Vizualization of TF_IDF scattering matrix as it evolves with
                   time. Each time step is a representation of the posts status
                   evolution that is fed into the RNN.
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plotScatterTime(time_serie):
    """ Given a time serie, makes an animation of the evolving data.
    Input : time_serie as a np bi-dimensionnal table.
    Output : void
    """

    print()

    x = []
    x2 = []
    x3 = []
    event = 3
    scale = 2.5

    # Creating a figure --------------------------------------------------------

    fig = plt.figure() # _init_ figure
    #ax=fig.add_axes([0,0,1,1])
    plt.xlabel("Timestamp sequence")
    plt.ylabel("TF-IDF for k-most important words")
    plt.title("Interval")

    for i in range(len(time_serie[event])) :
        max = len(time_serie[event][i])

        for j in range(len(time_serie[event][i])):
            x.append(j)

        plt.scatter(x, time_serie[event][i], color='r')
        filename='Plots/step'+str(i)+'.png'
        plt.savefig(filename, dpi=96)
        x = []
        # plt.scatter(x2, time_serie[event][count+1], color='b')
        # plt.scatter(x3, time_serie[event][count+2], color='y')

    plt.show()
