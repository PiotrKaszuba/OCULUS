import copy as cp
import math

import cv2
import numpy as np

import Code.Libraries.MyOculusImageLib as moil


# draws contour of one main circle area
def draw(pred, toDraw, morph_iter=0, threshold=127):
    w, h, c = moil.getWidthHeightChannels(pred)
    pred = np.uint8(pred * 255)
    ret, thresh = cv2.threshold(pred, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=morph_iter)

    eroded = cv2.erode(dilated, kernel, iterations=morph_iter)

    im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = 0.0
    ind = 0
    i = 0
    for cont in contours:
        try:
            temp = cv2.contourArea(cont) / cv2.arcLength(cont, True)
        except:
            temp = 0
        if temp > area:
            area = temp
            ind = i
        i += 1
    try:
        M = cv2.moments(contours[ind])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if toDraw is not None:
            cv2.drawContours(toDraw, contours, ind, (255, 255, 255), 2)
    except:
        cx = int(w / 2)
        cy = int(h / 2)


# seeks for main circle area and returns centerDiff metric for this circle
def centerDiff(pred, true=None, x=None, y=None, width=None, height=None, r=None, morph_iter=0, threshold=127,
               check=False, toDraw=None):
    assert true is not None or (
            x is not None and y is not None and width is not None and height is not None and r is not None)

    w, h, c = moil.getWidthHeightChannels(pred)
    pred = np.uint8(pred * 255)
    ret, thresh = cv2.threshold(pred, threshold, 255, cv2.THRESH_BINARY)
    if check:
        moil.show(thresh)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=morph_iter)
    if check:
        moil.show(dilated)
    eroded = cv2.erode(dilated, kernel, iterations=morph_iter)
    if check:
        moil.show(eroded)
    im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = 0.0
    ind = 0
    i = 0
    for cont in contours:
        try:
            temp = cv2.contourArea(cont) / cv2.arcLength(cont, True)
        except:
            temp = 0
        if temp > area:
            area = temp
            ind = i
        i += 1
    try:
        M = cv2.moments(contours[ind])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if toDraw is not None:
            cv2.drawContours(toDraw, contours, ind, (255, 255, 255), 2)
        if check:
            cop = cp.deepcopy(eroded)
            cv2.drawContours(cop, contours, ind, (0, 255, 255), 2)
            moil.show(cop)
    except Exception as e:
        cx = int(w / 2)
        cy = int(h / 2)

    if (x is None or y is None or width is None or height is None or r is None):

        width, height, chan = moil.getWidthHeightChannels(true)
        r = width / 10
        temp = true.dtype
        if not true.dtype == np.uint8:
            true = np.uint8(true * 255)
        ret, thresh = cv2.threshold(true, threshold, 255, cv2.THRESH_BINARY)
        if check:
            moil.show(thresh)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if check:
            cop = cp.deepcopy(thresh)
            cv2.drawContours(cop, contours, 0, (0, 255, 255), 2)
            moil.show(cop)
        try:
            M = cv2.moments(contours[0])
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        except:
            return 1

    w_scale = width / w
    h_scale = height / h
    cx = cx * w_scale
    cy = cy * h_scale

    dist = np.linalg.norm(np.asarray([cx, cy]) - np.asarray([x, y]))

    x1 = abs(width - x)

    y1 = abs(height - y)

    xtarg = int(max(x1, x) * 0.9 - 2 * r)
    ytarg = int(max(y1, y) * 0.9 - 2 * r)

    max_dist = math.sqrt(xtarg ** 2 + ytarg ** 2)
    met = (max_dist - dist) / max_dist

    print("Distance met: " + str(met) + ", distance: " + str(dist))

    return met


# returns binaryDiff for segmented image mask
def binaryDiff(pred, true, threshold=127, check=False):
    pred = np.uint8(pred * 255)
    if not true.dtype == np.uint8:
        true = np.uint8(true * 255)
    ret, thresh = cv2.threshold(pred, threshold, 255, cv2.THRESH_BINARY)
    ret, threshTrue = cv2.threshold(true, threshold, 255, cv2.THRESH_BINARY)

    unique, counts = np.unique(threshTrue, return_counts=True)
    trueC = dict(zip(unique, counts))

    diff = np.int16(threshTrue) - np.int16(thresh)
    if check:
        test = np.uint8((diff + 255) / 2)
        moil.show(test, other_im=[threshTrue, thresh])
    unique, counts = np.unique(diff, return_counts=True)
    diffC = dict(zip(unique, counts))

    try:
        TN = trueC[0]
    except:
        TN = 0
    try:
        TP = trueC[255]
    except:
        TP = 0

    try:
        FN = diffC[255]
    except:
        FN = 0
    try:
        FP = diffC[-255]
    except:
        FP = 0

    try:
        N = (TN - FP) / TN
    except:
        N = 1
    try:
        P = (TP - FN) / TP
    except:
        P = 1

    met = (N + P) / 2

    print("NP:" + str(met) + ", N: " + str(N) + ", P: " + str(P))
    return met


def customMetric(pred, true, check=False, toDraw=None):
    return (binaryDiff(pred, true, check=check) + centerDiff(pred, true, check=check, toDraw=toDraw)) / 2
