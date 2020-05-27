"""
File description : The first step of the preprocessing is removing special
                   characters from the input text. Especially ponctuation and
                   emojis.
"""


import numpy as np

def post_text_preprocess(temp,posts):
    """ Takes all posts array and size of each texts
    """
    temp = temp[0:np.count_nonzero(temp)] #interval 1
    post_text =[]
    text = ""
    for i in range(len(temp)):
        post_t = temp[i]
        p=next(item for item in posts if item["t"] == post_t)
        text =  str(p['text'])

        #Preprocessing
        proc_text =""
        #not_text=[" ","、","'","”","`","•","з","☆","^","」"," ∠",".","，","[","]","(",")","：",";","；","/","！","？","-","@",":","。",",","·","…","→","_","=","】","【","∀","~","～","*"]
        not_text=[" ","、","'","”","`","•","з","☆","^","」"," ∠",".","，","[","]","(",")","：",";","；","/","！","？","-","@",":","。",",","·","…","→","_","=","】","【","∀","~","～","*","🐶","😄","😝","😮","😋","💃","😃️","😍","😘","🌚","🌝","🐒","👧","🏀","😜","👄","💢","🍻","😊","🙏","🍊","✋","🌸","🌷","🌼","💐","🌹","🌺","🌻","🍄","🍃","🍂","🍁","👋","😭","💖","😎","😏","🌻","🍲","😌","💗","🔥","🌚","💯","👉","❤️","👸","🐱","🌻","✨","😂","🎵","👼","🇪🇸","➕","😉","😕","🚺","🐒","🐈","😞","💔","🍓","💛","💙","🔆","😖","🙋","😡","💪","💊","❌","🍔","🍰","🍭","🇷🇺","🍂","💚","☝️","🍬","💋","💃","😳","🎾","④","②","③","⑥","①","🇵🇱","👑","🎀","😇","😱","∩","😇","🙄","💤","🐛"]
        roman_alph = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        roman_alph_cap = [chr(x) for x in range(ord('A'), ord('Z') + 1)]
        for i in text:
            if ((isnumber(i)==False) and (i not in not_text) and (i not in roman_alph) and (i not in roman_alph_cap)):
                proc_text = proc_text+ str(i)
        if proc_text !="":
            post_text.append(proc_text)
    return post_text

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
