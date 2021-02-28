"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

from PIL import Image
from IPython.display import display
import PIL.ImageOps
import matplotlib.patches as patches
import os
import imutils
# from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np

def check_points_bas(image):
    if np.mean([np.mean(image[-5:-2,2:5]),np.mean(image[-5:-2,-5:-2])])>160:
        check=True
    else:
        check=False
    return check

def check_bar_horzontale_haut(image):
    if np.mean(image[1:7])>230:
        check=True
    else:
        check=False
    return check

def check_bar_central(image):
    if np.mean(image[:,20:30])>230:
        check=True
    else:
        check=False
    return check

def check_bar_4(image):
    if np.mean(image[1:-1,32:42])>230:
        check=True
    else:
        check=False
    return check

def check_no_point_central(image):
    if np.mean(image[23:28,23:28])<30:
        check=True
    else:
        check=False
    return check

def check_point_haut_gauche(image):
    if np.mean(image[2:8,3:5])>210:
        check=True
    else:
        check=False
    return check

def check_point_haut_milieu_droit(image):
    if np.mean(image[0:3,38:43])>100:
        check=True
    else:
        check=False
    return check
    

def check_no_point_central_gauche(image):
    if np.mean(image[12:32,2:10])<20:
        check=True
    else:
        check=False
    return check

def check_for_6(image):
    if np.mean(image[18:32,2:4])>220:
        check=True
    else:
        check=False
    return check

def check_for_9(image):
    if np.mean(image[30:38,2:8])<100:
        check=True
    else:
        check=False
    return check

def check_bug_not_9(image):
    if not np.mean([np.mean(image[:2,:10]),np.mean(image[:2,-10:])])<30:
        check=True
    else:
        check=False
    return check

def check_point_central_bas(image):
    if np.mean(image[30:35,20:30])>200:
        check=True
    else:
        check=False
    return check

def predict(img_number,coord):
    [x1, y1, x2, y2] = coord
    points_bas = check_points_bas(img_number)
    bar_centre_vertical = check_bar_central(img_number)
    bar_4 = check_bar_4(img_number)
    point_central_bas = check_point_central_bas(img_number)
    bar_horzontale_haut = check_bar_horzontale_haut(img_number)
    point_haut_gauche = check_point_haut_gauche(img_number)
    no_point_central_gauche = check_no_point_central_gauche(img_number)
    no_point_central = check_no_point_central(img_number)
    point_6 = check_for_6(img_number)
    point_9 = check_for_9(img_number)
    point_haut_milieu_droit = check_point_haut_milieu_droit(img_number)
    bug_not_9 = check_bug_not_9(img_number)
    bugged=0
    if points_bas==True:
        if bar_centre_vertical==True:
            prediction=1
        else:
            prediction=2
    elif bar_4==True and point_central_bas==True:
        prediction=4
        # plt.figure()
        # plt.imshow(img_number)
    elif bar_horzontale_haut==True:
        prediction=7
    elif no_point_central_gauche==True:
        prediction=3
    elif point_haut_gauche==True or point_haut_milieu_droit==True:
        # plt.figure()
        # plt.imshow(img_number)
        prediction=5
    elif no_point_central==True:
        prediction=0
    elif point_6==True:
        prediction=6
    elif point_9==True:
        if bug_not_9==True:
            bugged=1
        prediction=9
    else:
        prediction=8
            
    return prediction,bugged

def make_prediction(img_number,coord):
    prediction, bugged  = predict(img_number,coord)
    plt.figure()
    while bugged==1 and prediction==9:
        img_number = cv2.resize(img_number[1:], (50,50))
        prediction,bugged = predict(img_number,coord)
        plt.imshow(img_number)
        plt.title(prediction)
    return prediction

def image_processing(image):
    # pre-process the image by resizing it, converting it to
    image = imutils.resize(image, height=2000)
    image = cv2.resize(image, (2000,2000), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 180, 255,cv2.THRESH_TOZERO)[1]
    thresh = cv2.threshold(thresh, 0, 255,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    image_mod = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    image_mod = 255-thresh
    return image_mod

def resize_with_coord(coord,image):
    xmin,ymin,_,_ = coord.min(axis=0)
    _,_,xmax,ymax = coord.max(axis=0)
    image_mod = image[ymin:ymax,xmin:xmax]
    coord = coord-[xmin,ymin,xmin,ymin]
    s1 = image_mod.shape
    image_mod = cv2.resize(image_mod,(2000,2000))
    s2 = image_mod.shape
    ratio_y = s2[0]/s1[0]
    ratio_x = s2[1]/s1[1]
    coord = np.array(np.round(coord*[ratio_x,ratio_y,ratio_x,ratio_y]),dtype=int)
    return coord,image_mod

def find_number(image):
    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    coord = []
    # fig,ax=plt.subplots()
    # plt.imshow(image)
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        #Remove Bingo
        bingo_c = (y<=800 or y>=1000)
        #Remove free
        f1 = not ((y>450 and y<550) and (x>450 and x<550))
        f2 = not ((y>1300 and y<1500) and (x>450 and x<550))
        f3 = not ((y>450 and y<550) and (x>1450 and x<1500))
        f4 = not ((y>1300 and y<1500) and (x>1450 and x<1500))
        if (w >= 40 and w <= 80) and (h >= 85 and h <= 100) and y >=50 and x>=50 and bingo_c and f1 and f2 and f3 and f4:
            # rect= patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='g', facecolor='none')
            # ax.add_patch(rect)
            coord.append([x, y, w+x, h+y])
    coord=np.array(coord)
    return coord

def find_number_2(image):
    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    coord = []
    # fig,ax=plt.subplots()
    # plt.imshow(image)
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        #Remove Bingo
        bingo_c = (y<=800 or y>=1100)
        #Remove free
        f1 = not ((y>300 and y<500) and (x>320 and x<550))
        f2 = not ((y>1500 and y<1650) and (x>320 and x<550))
        f3 = not ((y>300 and y<500) and (x>1450 and x<1600))
        f4 = not ((y>1500 and y<1650) and (x>1450 and x<1600))
        bar_vert = not (x>930 and x<1050)
        if (w >= 40 and w <= 100) and bar_vert and (h >= 85 and h <= 160) and bingo_c and f1 and f2 and f3 and f4:
            # rect= patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='g', facecolor='none')
            # ax.add_patch(rect)
            coord.append([x, y, w+x, h+y])
    coord=np.array(coord)
    return coord

def find_numbers_and_resize(image):
    coord = find_number(image)
    coord,image_mod = resize_with_coord(coord,image)
    coord = find_number_2(image_mod)
    return coord, image_mod
