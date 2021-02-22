"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import matplotlib.pyplot as plt

# load the example image
image = cv2.imread('/Users/JP/Documents/Crosser-le-Maire/Cartes_bingo_1/151752913_809760336289159_6298217597609049271_n.jpg')

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=2000)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    #Remove Bingo
    bingo_c = (y<=770 or y>=1000)
    #Remove free
    f1 = not ((y>450 and y<550) and (x>450 and x<550))
    f2 = not ((y>1300 and y<1550) and (x>450 and x<550))
    f3 = not ((y>450 and y<550) and (x>1450 and x<1550))
    f4 = not ((y>1300 and y<1550) and (x>1450 and x<1550))
    if (w >= 40 and w <= 80) and (h >= 85 and h <= 100) and y >=50 and bingo_c and f1 and f2 and f3 and f4:
        crop_img = thresh[y:y+h, x:x+w]
        plt.figure()
        cv2.rectangle(thresh, (x, y), (w+x, h+y), (255,0,0), 2)
        print((w, h))
        digitCnts.append(c)
        