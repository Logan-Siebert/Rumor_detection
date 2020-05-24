"""
File description : Building time serie structure for posts.



"""

import numpy as np
import math as m

def equipartition(posts, l):
    """
    """
    #equipartition in intervals of length l
    U = np.zeros((1, len(posts)), dtype=int)
    u = np.zeros(len(posts), dtype=int)
    i=0
    while(np.count_nonzero(U)<len(posts) and i<1000000):
        cond= (posts <l*(i+1)+posts[0])*(posts >= l*i+posts[0])
        u=posts[cond]
        if u.shape[0]!=0: #Non empty intervals, rows: intervals, colums: posts
            u= np.hstack((u,np.zeros(len(posts)-len(u),dtype=int)))
            U=np.vstack((U,u))
        i=i+1
    if(i==1000000):
        U = np.zeros((len(posts),1), dtype=int)
        for j in range(len(posts)):
            U[j]=posts[j]

    count = np.count_nonzero(U,1)
    idx = max(count)
    return U[1:U.shape[0],0:idx]

def time_series_const(N,posts):  # For 1 event
    """
    """
    L = int(max(posts)) #entire event timeline
    l= int(round(L/N))  #intervals lengths
    U = equipartition(posts,l) #rows: intervals, colums: posts

    n_intervals_prev=0
    n_intervals=U.shape[0]
    while(U.shape[0]<N and n_intervals > n_intervals_prev): # >= to force timesries >=N but too long too process
        #second condition to limit time processing
        l=int(m.floor(l/2))
        U = equipartition(posts,l)
        n_intervals_prev=n_intervals
        n_intervals = U.shape[0]
    return U


def events_time_series(event_related_posts,N): #for all events
    """
    """
    time_series =[]
    for i in range(event_related_posts.shape[0]):
        print("event %d" %(i+1))
        posts=event_related_posts[i]
        mask = posts!=0
        posts = posts[mask]
        start = posts[0]
        posts = posts - posts[0] + 1
        t= time_series_const(N,posts)
        t[np.nonzero(t)] = t[np.nonzero(t)] + start - 1
        time_series.append(t)
    return time_series
