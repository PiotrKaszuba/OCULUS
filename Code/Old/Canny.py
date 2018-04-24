import cv2
import numpy as np
import copy as cp

def callback(x):
    pass

def show(im):
    while cv2.waitKey(30) != ord('q'):
        cv2.imshow("win", im)



cv2.namedWindow("win")

name = "1"

im = cv2.imread("../../Images/test/"+name+".jpg",0)

im=cv2.resize(im, (0,0), fx=0.2, fy=0.2)
#show(im)
#im = cv2.equalizeHist(im)
show(im)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(35,35))
im = clahe.apply(im)
#show(im)
im=cv2.GaussianBlur(im,(5,5),2,0)

show(im)
im = cv2.Canny(im,100,200)



show(im)