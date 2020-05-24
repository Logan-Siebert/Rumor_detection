"""
File description : TF_IDF extraction from preprocessed texts. Simple weighting
                   methods.
                   Term frequency–inverse document frequency

                   TF_IDF = term frequency * inverse document frequency.
"""

import math as m

def tfidf(post_text):
    """ Takes concatenated textual content, returns TF_IDF values.
    Input : post_text (String)
    Output : TF-IDF (float)
    """

    # Document frequency (len(DF)= total number of words in all docs)
    DF = {}
    k=0
    for elem in post_text:
        for i in range(len(elem)):
            w = elem[i]
            try:
                if(k not in DF[w]):
                    DF[w].add(k)
            except:
                DF[w] = {k}
        k=k+1


    for i in DF:
        DF[i]=len(DF[i])  #dict: number of docs where word i appear

    df={}
    TF=[] # Word document term frequency
    for elem in post_text:
       for i in range(len(elem)):
           w = elem[i]
           try:
               df[w].add(i)
           except:
               df[w] = {i}
       for j in df:
           df[j]=len(df[j])/len(df)
       TF.append(df)
       df={}

    #Inverse document frequency : IDF
    IDF={}
    for word in DF:
        if(len(post_text)>1):
            IDF[word]= m.log10(len(post_text)/DF[word])
        else:
            IDF[word]=1

    tfidf = {}
    TF_IDF=[]
    for doc in TF:
       for word in doc:
           tfidf[word] = doc[word]*IDF[word]
       TF_IDF=TF_IDF+list(tfidf.values())
       tfidf = {}

    return TF_IDF
