import cv2
import numpy as np
import math
import copy
#checks if given point x,y is inside the circle of r radius
#circle center a,b is r,r when not given
def in_circle(x,y,r, a=None, b=None):
    if a==None:
        a=r
    if b==None:
        b=r
    if( (x-a)**2 + (y-b)**2 <= r**2 ):
        return True;
    else:
        return False;

#creates a kernel with given value(default = 1) of given data type in circular area from middle
#with radius of r and zeros in the rest of positions, kernel sizes are 2*r+1 for symmetric and centered
def circle_kernel(r, v=1 , type=np.uint8):
    #r must be integer and higher than 0
    assert r > 0 and r%1==0
    kernel = np.zeros((2*r+1,2*r+1), dtype=type)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if(in_circle(i,j,r,r,r)):
                kernel[i,j]=v
    return kernel


def circle_filter_2d(im, r):
    im = cv2.filter2D(im, 5, circle_kernel(r))
    im /= (r ** 2 * 3.14)
    im = cv2.convertScaleAbs(im)

    return im


def minus_mean(im_subtracted, im_mean=None):
    if im_mean == None:
        im_mean=im_subtracted


    m = np.mean(im_mean)
    temp = np.int16(im_subtracted) - np.int16(m)
    temp[temp < 0] = 0
    return np.uint8(temp)

def trackback_callback(x):
    pass


def read_and_size( name, path='Images/', extension='.jpg', scale=0.3, mode=0):
    im = cv2.imread(path+name+extension, mode)
    im = cv2.resize(im, (0,0), fx=scale, fy=scale)
    return im

def read_and_size_with_copy( name, path='Images/', extension='.jpg', scale=0.3, mode=0):
    im = cv2.imread(path+name+extension, mode)
    im = cv2.resize(im, (0,0), fx=scale, fy=scale)
    im_copy = copy.deepcopy(im)
    return im, im_copy



im,im_org = read_and_size_with_copy('0')
im_other = copy.deepcopy(im)


#im_other = minus_mean(im_other)

r=25


im = circle_filter_2d(im, r)
#im = cv2.equalizeHist(im)
im_minus_itself= copy.deepcopy(im)
#im_minus_org = copy.deepcopy(im)

im_minus_itself = minus_mean(im_minus_itself)
#im_minus_org = minus_mean(im_minus_org, im_org)


imi = copy.deepcopy(im_minus_itself)
#imo = copy.deepcopy(im_minus_org)
cv2.namedWindow('minus_itself')
cv2.createTrackbar('threshold','minus_itself', 0, 255, trackback_callback)
cv2.createTrackbar('clip','minus_itself', 1, 100, trackback_callback)
cv2.createTrackbar('tile','minus_itself', 1, 100, trackback_callback)
while(cv2.waitKey(30) != ord('q')):
    cv2.imshow('win',im)


    clahe = cv2.createCLAHE(clipLimit=cv2.getTrackbarPos('clip','minus_itself')*0.05, tileGridSize=(cv2.getTrackbarPos('tile','minus_itself'), cv2.getTrackbarPos('tile','minus_itself')))
    im_oth = clahe.apply(im_other)
    cv2.imshow('other', im_oth)
    ret, thresh_imi = cv2.threshold(imi, cv2.getTrackbarPos('threshold','minus_itself'), 255, cv2.THRESH_BINARY)
    cv2.imshow('minus_itself', im_minus_itself)
    cv2.imshow('thresh', thresh_imi)
    #cv2.imshow('minus_org', im_minus_org)
    cv2.imshow('orginal', im_org)