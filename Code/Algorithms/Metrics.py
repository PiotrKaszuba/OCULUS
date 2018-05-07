import cv2
import numpy as np
import Code.Libraries.MyOculusLib as mol
import copy as cp
import math
from collections import Counter
def centerDiff(pred, true = None, x=None, y=None, width=None, height=None, r=None, morph_iter = 8, threshold = 127, check=False):
    assert true is not None or (x is not None and y is not None and width is not None and height is not None and r is not None)
    w,h,c = mol.getWidthHeightChannels(pred)
    pred = np.uint8(pred*255)
    ret, thresh = cv2.threshold(pred, threshold,255, cv2.THRESH_BINARY)
    if check:
        mol.show(thresh)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=morph_iter)
    if check:
        mol.show(dilated)
    eroded = cv2.erode(dilated, kernel, iterations=morph_iter)
    if check:
        mol.show(eroded)
    im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    area = 0.0
    ind = 0
    i=0
    for cont in contours:
        try:
            temp = cv2.contourArea(cont) / cv2.arcLength(cont, True)
        except:
            temp = 0
        if temp > area:
            area = temp
            ind = i
        i+=1
    M = cv2.moments(contours[ind])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    if check:
        cop = cp.deepcopy(eroded)
        cv2.drawContours(cop, contours, ind, (0, 255, 255), 2)
        mol.show(cop)

    if( x is None or y is None or width is None or height is None or r is None):
        width, height, chan = mol.getWidthHeightChannels(true)
        r = width/10
        true = np.uint8(true*255)
        ret, thresh = cv2.threshold(true, threshold, 255, cv2.THRESH_BINARY)
        if check:
            mol.show(thresh)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if check:
            cop = cp.deepcopy(thresh)
            cv2.drawContours(cop, contours, 0, (0, 255, 255), 2)
            mol.show(cop)
        M = cv2.moments(contours[0])
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])


    w_scale = width/w
    h_scale = height/h
    cx = cx*w_scale
    cy = cy*h_scale

    dist = np.linalg.norm(np.asarray([cx, cy]) - np.asarray([x, y]))

    x1 = abs(width - x)

    y1 = abs(height - y)


    xtarg = max(x1,x) - r
    ytarg = max(y1,y) - r




    max_dist = math.sqrt(xtarg**2 + ytarg**2)
    return (max_dist-dist)/max_dist

def binaryDiff(pred, true, threshold=127):
    pred = np.uint8(pred * 255)
    true = np.uint8(true * 255)
    ret, thresh = cv2.threshold(pred, threshold, 255, cv2.THRESH_BINARY)
    ret, threshTrue = cv2.threshold(true, threshold, 255, cv2.THRESH_BINARY)

    unique, counts = np.unique(threshTrue, return_counts=True)
    trueC = dict(zip(unique, counts))

    diff = np.int16(threshTrue) - np.int16(thresh)

    unique, counts = np.unique(diff, return_counts=True)
    diffC = dict(zip(unique, counts))

    try:
        TN=  trueC[0]
    except:
        TN=0
    try:
        TP=  trueC[255]
    except:
        TP=0


    try:
        FN=  diffC[255]
    except:
        FN=0
    try:
        FP=  diffC[-255]
    except:
        FP=0

    try:
        N = (TN-FP)/TN
    except:
        N = 1
    try:
        P = (TP-FN)/TP
    except:
        P=1
    return (N+P)/2

def customMetric(pred,true):
    return (binaryDiff(pred,true)+centerDiff(pred,true))/2
