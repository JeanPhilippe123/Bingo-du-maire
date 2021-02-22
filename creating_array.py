"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

def find_card(coord):
    card=[]
    for i in range(0,len(coord)):
        x1,y1,x2,y2 = coord[i]
        if x1<1000 and y1<1000:
            card.append(1)
        elif x1 >1000 and y1<1000:
            card.append(2)
        elif x1 <1000 and y1>1000:
            card.append(3)
        elif x1>1000 and y1>1000:
            card.append(4)
        else:
            card.append(np.nan)
    return card

def find_position(coord):
    l=[]
    c=[]
    for i in range(0,len(coord)):
        position = np.array(list(map(math.floor,coord[i]/50)))*50
        x1,y1,x2,y2 = position
        lines = np.arange(200,2001,200)
        for i in range(0,len(lines)):
            if y1<lines[i]:
                l.append(i)
                break
        columns = lines
        for i in range(0,len(columns)):
            if x1<columns[i]:
                c.append(i)
                break
    return np.array(l),np.array(c)

def create_array(lc,coord,prediction):
    #Construct array
    array_card = np.zeros([10,10])
    
    value = list(itertools.combinations(lc, 2))
    index = list((i,j) for ((i,_),(j,_)) in itertools.combinations(enumerate(lc), 2))
    for i in range(0,len(value)):
        a,b = value[i]
        ind = index[i]
        if (a == b).all():
            #Check which is the dizaine criss
            if coord[ind[0]][0] < coord[ind[1]][0]:
                array_card[tuple(lc[i])] =  prediction[ind[0]]*10+prediction[ind[1]]
            else: 
                array_card[tuple(lc[i])] =  prediction[ind[0]]+prediction[ind[1]]*10
            break
    return array_card