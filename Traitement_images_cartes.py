"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

from PIL import Image
from IPython.display import display
import PIL.ImageOps
import os
import imutils
# from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np

def check_bar_horzontale_bas(image):
    if np.mean(image[-7:-1,1:-1])>230:
        check=True
    else:
        check=False
    return check

def check_bar_horzontale_haut(image):
    if np.mean(image[1:7])>240:
        check=True
    else:
        check=False
    return check

def check_bar_central(image):
    if np.mean(image[:,20:30])>240:
        check=True
    else:
        check=False
    return check

def check_bar_4(image):
    if np.mean(image[:,32:42])>240:
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
    if np.mean(image[2:5,3:6])>230:
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

def make_prediction(image,coord):
    [x1, y1, x2, y2] = coord
    img_number = image
    bar_bas_horizontal = check_bar_horzontale_bas(img_number)
    bar_centre_vertical = check_bar_central(img_number)
    bar_4 = check_bar_4(img_number)
    bar_horzontale_haut = check_bar_horzontale_haut(img_number)
    point_haut_gauche = check_point_haut_gauche(img_number)
    no_point_central_gauche = check_no_point_central_gauche(img_number)
    no_point_central = check_no_point_central(img_number)
    point_6 = check_for_6(img_number)
    point_9 = check_for_9(img_number)
    if bar_bas_horizontal==True:
        if bar_centre_vertical==True:
            prediction=1
        else:
            prediction=2
    elif bar_4==True:
        prediction=4
    elif bar_horzontale_haut==True:
        prediction=7
    elif no_point_central_gauche==True:
        prediction=3
    elif point_haut_gauche==True:
        prediction=5
    elif no_point_central==True:
        prediction=0
    elif point_6==True:
        prediction=6
    elif point_9==True:
        prediction=9
    else:
        prediction=8
            
    return prediction

def image_processing(image):
    # pre-process the image by resizing it, converting it to
    image = imutils.resize(image, height=2000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    image_mod = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return image_mod

def find_number(thresh):
    
    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    coord = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        #Remove Bingo
        bingo_c = (y<=800 or y>=1000)
        #Remove free
        f1 = not ((y>450 and y<550) and (x>450 and x<550))
        f2 = not ((y>1300 and y<1550) and (x>450 and x<550))
        f3 = not ((y>450 and y<550) and (x>1450 and x<1550))
        f4 = not ((y>1300 and y<1550) and (x>1450 and x<1550))
        if (w >= 40 and w <= 80) and (h >= 85 and h <= 100) and y >=50 and bingo_c and f1 and f2 and f3 and f4:
            # plt.figure()
            # cv2.rectangle(thresh, (x, y), (w+x, h+y), (255,0,0), 2)
            coord.append([x, y, w+x, h+y])
    return np.array(coord)