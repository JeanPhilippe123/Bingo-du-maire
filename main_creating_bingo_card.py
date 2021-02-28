"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import shutil
import imutils
import math
import itertools
import glob
import PIL

import Traitement_images_cartes as ti

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
        lines = [125,300,500,700,1000,1250,1450,1600,1800,2000]
        for i in range(0,len(lines)):
            if y1<lines[i]:
                l.append(i)
                break
        columns = [125,350,550,750,1000,1200,1450,1600,1800,2000]
        for i in range(0,len(columns)):
            if x1<columns[i]:
                c.append(i)
                break
    return np.array(l),np.array(c)

def create_array(line,column,coord,prediction):
    array = np.zeros((10,10))
    for i,array_line in enumerate(array):
        for j,array_number in enumerate(array_line):
            if ((i==2)|(i==7) and (j==2)|(j==7)):
                continue
            filt = (line == i) & (column == j)
            if len(line[filt])==1:
                index = np.where(filt)[0][0]
                array[i,j] = prediction[index]
            else:
                index_1 = np.where(filt)[0][0]
                index_2 = np.where(filt)[0][1]
                if coord[index_1][0] < coord[index_2][0]:
                    array[i,j] = 10*prediction[index_1]+1*prediction[index_2]
                else:
                    array[i,j] = 10*prediction[index_2]+1*prediction[index_1]
    card_1 = np.expand_dims(array[:5,:5],axis=0)
    card_2 = np.expand_dims(array[5:,:5],axis=0)
    card_3 = np.expand_dims(array[:5,5:],axis=0)
    card_4 = np.expand_dims(array[5:,5:],axis=0)
    card = np.concatenate([card_1,card_2,card_3,card_4])
    return  card

main_directory = '/Users/JP/Documents/Crosser-le-Maire'
directory = main_directory+'/Photos_cartes_bingo/'
Images_path = glob.glob(directory+'*.jpg')[:]
n=0
m=0

for i in range(0,len(Images_path)):
    try:
        #Load the example image
        image = cv2.imread(Images_path[i])
        image_card_bingo = PIL.Image.open(Images_path[0])
        image_mod = ti.image_processing(image)
        coord,image_mod = ti.find_numbers_and_resize(image_mod)
        # plt.imshow(image_mod)
        # coord,image_mod = ti.resize_with_coord(coord,image_mod)
        x1,y1,x2,y2 = coord.transpose()
        img_number = []
        prediction = []
        for j in range(0,len(coord)):
            crop_img = image_mod[y1[j]:y2[j],x1[j]:x2[j]]
            img_number.append(cv2.resize(crop_img, (50,50)))
            prediction.append(ti.make_prediction(img_number[j],coord[j]))
        
        #Divide card
        card = find_card(coord)
        #Find line and columns
        line,columns = find_position(coord)
        
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(image_mod)
        # for i in range(0,len(coord)):
        #     rect= patches.Rectangle((x1[i],y1[i]),x2[i]-x1[i],y2[i]-y1[i],linewidth=1, edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(x1[i],y1[i],str(prediction[i]),color='r')    
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(image_mod)
        # for i in range(0,len(coord)):
        #     rect= patches.Rectangle((x1[i],y1[i]),x2[i]-x1[i],y2[i]-y1[i],linewidth=1, edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(x1[i],y1[i],str(line[i]),color='r')
        # fig,ax=plt.subplots(1,1)
        # ax.imshow(image_mod)
        # for i in range(0,len(coord)):
        #     rect= patches.Rectangle((x1[i],y1[i]),x2[i]-x1[i],y2[i]-y1[i],linewidth=1, edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(x1[i],y1[i],str(columns[i]),color='r')    
        
        #Create array
        card_bingo = create_array(line,columns,coord,prediction)
        np.save(main_directory+'/Fichiers_npy/'+str(i)+'.npy',[card_bingo,image_card_bingo])
    except:
        # print(i,Images_path[i],'\n')
        n+=1
    if (card_bingo>=90).any():
        # print(i,Images_path[i],'\n')
        m+=1
        