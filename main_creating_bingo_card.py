"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Traitement_images_cartes as ti
import cv2
import os
import shutil
import imutils
import math
import itertools
import glob

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

directory = '/Users/JP/Documents/Crosser-le-Maire/Cartes_bingo_1/'
Images_path = glob.glob(directory+'*.jpg')[2:4]
if not os.path.exists(directory+'Numbers'):
    os.mkdir(directory+'Numbers')

for i in range(0,len(Images_path)):
    #Load the example image
    image = cv2.imread(Images_path[i])
    image_mod = ti.image_processing(image)
    coord = ti.find_number(image_mod)
    x1,y1,x2,y2 = coord.transpose()
    plt.figure()
    # plt.imshow(image_mod[y1[0]:y2[0],x1[0]:x2[0]])
    img_number = []
    prediction = []
    for i in range(0,len(coord)):
        crop_img = image_mod[y1[i]:y2[i],x1[i]:x2[i]]
        img_number.append(cv2.resize(crop_img, (50,50)))
        prediction.append(ti.make_prediction(img_number[i],coord[i]))

    #Create array
    #Divide card
    # card = find_card(coord)
    #Find line and columns
    # line,columns = find_position(coord)
    # lc = np.array([line,columns]).transpose()
    
    # print(create_array(lc,coord,prediction))
    
    #Create array
    # array = find_array()
    fig,ax=plt.subplots(1,1)
    ax.imshow(image_mod)
    for i in range(0,len(coord)):
        rect= patches.Rectangle((x1[i],y1[i]),x2[i]-x1[i],y2[i]-y1[i],linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1[i],y1[i],str(prediction[i]),color='r')
    
    