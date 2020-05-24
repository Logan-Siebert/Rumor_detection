"""
File description : Complete dataset visualization (character-based Weibo data)
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plotEventsExpecteValue(time_serie):
    """ Given a time serie, makes an animation of the evolving data.
    Input : time_serie as a np bi-dimensionnal table.
    Output : void
    """

    scale = 2.5
    maxTime = 50
    # Building data from each step, for each event

    x = []
    eV = []
    var = []
    # Creating a figure --------------------------------------------------------

    fig = plt.figure() # _init_ figure
    plt.xlabel("Event order position (indice)")
    plt.ylabel("Mean TF-IDF value for all words")
    plt.title("All Weibo dataset representation as TF-IDF (Expected value)")

    t = 0

    # One has to impose maximum time serie after which the timeloop breaks.
    while t < maxTime :
        for i in range(len(time_serie)) :   #Sweeping through all events
            moment = 0
            count = 1;

            if(t < len(time_serie[i])) :
                for j in range(len(time_serie[i][t])) :
                    moment += time_serie[i][t][j]
                    count+=1

            moment = moment/count
            eV.append(moment)
            x.append(i/scale)

        #Scattering at time t
        plt.scatter(x, eV, color='r', s=1)
        filename='Plots/E/step'+str(t)+'.png'
        plt.savefig(filename, dpi=200)
        x = []
        eV = []
        t += 1

    plt.show()
