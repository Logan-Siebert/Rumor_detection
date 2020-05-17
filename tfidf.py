import math as m

def tfidf(post_text):
    DF = {} # Document frequency (len(DF)= total number of words in all docs)
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

    #IDF
    IDF={}
    for word in DF:
        IDF[word]= m.log10(len(DF)/DF[word])
    
    tfidf = {}
    TF_IDF=[]
    for doc in TF:
       for word in doc:
           tfidf[word] = doc[word]*IDF[word]
       TF_IDF=TF_IDF+list(tfidf.values())
       tfidf = {}
        
    return TF_IDF