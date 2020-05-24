"""
File description : Extracting textual data from the labeled events
                   Formatting : Weibo.txt
                                4664 labled events
                                each line -> one event with ids of relevant posts
                                (event_id, label, post_ids)

                   Content of the posts in .json format (event_id.json)
"""

import numpy as np

def extract_dataset(event_ids_list, labels_list, filename, n_ev):
    """ Given empty event_ids, label_list, the data textfile and the dataset size, returns
        Properly formatted posts-event matrix, label vector and relevant ids vector.

    Input : event_ids_list : np array (size = example size n_ev)
            labels_list : np array
            filename : String to path of file
            n_ev : size of the data
    """

    # fill event ids ans labels arrays. return event related posts arry
    # each row is one event and each column a post
    dataset = open(filename, "r")
    lines = dataset.readlines()
    # max number of posts per event
    maxlen=0
    for line in lines:
        elems = line.split()
        if len(elems)-2 > maxlen:
            maxlen = len(elems)-2
    # 0 = no post
    event_related_posts=np.zeros((n_ev,maxlen),dtype=int)
    i=0
    for line in lines:
        elems = line.split()
        event_id = elems[0] # 1st elem is the event id
        label= elems[1]     # 2nd elem is the label
        event_ids_list[i]=(event_id[4:len(event_id)])
        labels_list[i] = label[6]
        # event_related_posts[i,0:len(elems)-2]=elems[2:len(elems)] #posts per event
        # event_related_posts[i]=np.sort(event_related_posts[i])
        i=i+1
    dataset.close()
    return event_related_posts
