import math

import cv2
import numpy as np

import Code.Libraries.MyOculusImageLib as moil

def atrophyRate(img, threshold = 127):
    img = moil.getBinaryThreshold(img, threshold)
    unique, counts = np.unique(img, return_counts=True)
    try:
        count  = dict(zip(unique, counts))[255]
    except:
        count = 0
    return count / 20576




# draws contour of one main circle area
def draw(pred, toDraw, morph_iter=0, threshold=127, thickness=2):
    thresh = moil.getBinaryThreshold(pred, threshold)

    closed = moil.morphMultiClosing(thresh, morph_iter)

    contour = moil.selectBiggerCircularContour(closed)

    if toDraw is not None and contour is not None:
        cv2.drawContours(toDraw, [contour], -1, (0, 0, 255), thickness)


def getCenter(pred, widthRef, heightRef, morph_iter=0, threshold=127):
    thresh = moil.getBinaryThreshold(pred, threshold)

    closed = moil.morphMultiClosing(thresh, morph_iter)
    contour = moil.selectBiggerCircularContour(closed)
    w, h, c = moil.getWidthHeightChannels(pred)
    try:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    except Exception as e:
        print("No contour detected! Guessing for center...")
        cx = int(w / 2)
        cy = int(h / 2)
    w_scale = widthRef / w
    h_scale = heightRef / h
    cx = cx * w_scale
    cy = cy * h_scale
    return cx, cy


# seeks for main circle area and returns centerDiff metric for this circle
def centerDiff(pred, true=None, x=None, y=None, width=None, height=None, r=None, morph_iter=0, threshold=127,
               toDraw=None, retDist=False):
    assert true is not None or (
            x is not None and y is not None and width is not None and height is not None and r is not None)

    thresh = moil.getBinaryThreshold(pred, threshold)

    closed = moil.morphMultiClosing(thresh, morph_iter)
    contour = moil.selectBiggerCircularContour(closed)
    if toDraw is not None and contour is not None:
        cv2.drawContours(toDraw, [contour], -1, (255, 255, 255), 1)

    w, h, c = moil.getWidthHeightChannels(pred)
    try:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    except Exception as e:
        print("No contour detected! Guessing for center...")
        cx = int(w / 2)
        cy = int(h / 2)

    if (x is None or y is None or width is None or height is None or r is None):

        width, height, chan = moil.getWidthHeightChannels(true)
        r = width / 10

        thresh = moil.getBinaryThreshold(true, threshold)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            M = cv2.moments(contours[0])
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        except:
            print("Bad Ground-truth! Mask in center...")
            x = int(w / 2)
            y = int(h / 2)

    w_scale = width / w
    h_scale = height / h
    cx = cx * w_scale
    cy = cy * h_scale

    dist = np.linalg.norm(np.asarray([cx, cy]) - np.asarray([x, y]))

    if retDist:
        return dist

    maxDist = np.linalg.norm(np.asarray([w / 2, h / 2]) - np.asarray([x, y]))

    DistanceMetric = 1 - dist / maxDist

    CrossLength = math.sqrt(width ** 2 + height ** 2)

    DistanceToCross = dist / CrossLength

    print("Distance Metric: " + str(DistanceMetric) + ", Relative Distance: " + str(
        DistanceToCross) + ", Distance: " + str(dist))

    return DistanceMetric


# returns binaryDiff for segmented image mask
def binaryDiff(pred, true, threshold=127, globalCount=False):
    thresh = moil.getBinaryThreshold(pred, threshold)
    threshTrue = moil.getBinaryThreshold(true, threshold)

    unique, counts = np.unique(threshTrue, return_counts=True)
    trueC = dict(zip(unique, counts))

    diff = np.int16(threshTrue) - np.int16(thresh)

    unique, counts = np.unique(diff, return_counts=True)
    diffC = dict(zip(unique, counts))

    try:
        Negatives = trueC[0]
    except:
        Negatives = 0
    try:
        Positives = trueC[255]
    except:
        Positives = 0

    try:
        FN = diffC[255]
    except:
        FN = 0
    try:
        FP = diffC[-255]
    except:
        FP = 0

    TP = Positives - FN
    TN = Negatives - FP

    try:
        Specifity = TN / Negatives
    except:
        Specifity = 1
    try:
        Sensitivity = TP / Positives
    except:
        Sensitivity = 1


    Accuracy = (TP + TN) / (Positives + Negatives)
    Jouden = Sensitivity + Specifity - 1
    print("Youden Index: " + str(Jouden) + ", Sensivity: " + str(Sensitivity) + ", Specifity: " + str(
        Specifity) + ", Accuracy: " + str(Accuracy) )
    if globalCount:
        return [TP, FN, FP, TN]
    return Jouden

def globals(confusion_matrix):
    TP, FN, FP, TN = confusion_matrix
    try:
        precision = TP / (TP + FP)
        Sensitivity = TP /(TP + FN)
        Specificity = TN / (TN + FP)
        Youden = Sensitivity + Specificity - 1
        pseudo_dice = 2*precision*Sensitivity/(precision+Sensitivity)
        pseudo_jaccard = precision * Sensitivity / ((precision + Sensitivity) - (precision * Sensitivity))
    except:
        pseudo_dice = -100
        pseudo_jaccard = -100

    return [Youden, pseudo_jaccard, pseudo_dice]


def customMetric(pred, true, toDraw=None, metrics=['distance', 'youden', 'jaccard', 'dice'], globalCount=False, threshold = 127):
    ret = []
    glob = [0]*4
    if globalCount:
        glob = binaryDiff(pred, true, threshold=threshold, globalCount=globalCount)
    if 'atrophy' in metrics:
        print("Predicted Atrophy rate: " +str(atrophyRate(pred, threshold=threshold)))
        print("True Atrophy rate: " + str(atrophyRate(true, threshold=threshold)))
    if 'distance' in metrics:
        Distance = centerDiff(pred, true, threshold=threshold, toDraw=toDraw, retDist=True)
        ret.append(Distance)
    if 'youden' in metrics:
        Jouden = binaryDiff(pred, true, threshold=threshold)
        ret.append(Jouden)
    if 'jaccard' in metrics:
        jaccard = jaccard_index(true, pred, threshold=threshold)
        ret.append(jaccard)
    if 'dice' in metrics:
        dice = dice_coefficient(true, pred, threshold=threshold)
        ret.append(dice)
    print("-------------------------")
    if 'distance' in metrics:
        print("Distance Improvement: " + str(Distance))
    if 'youden' in metrics:
        print("Youden Index: " + str(Jouden))
    if 'jaccard' in metrics:
        print("Jaccard Index: " + str(jaccard))
    if 'dice' in metrics:
        print("Dice Sorensen coefficient: " + str(dice))
    print("-------------------------")
    return [ret, glob]


def jaccard_index(ground_truth, prediction, threshold=127):
    ground_truth = moil.getBinaryThreshold(ground_truth, threshold)
    prediction = moil.getBinaryThreshold(prediction, threshold)
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    jaccard = np.sum(intersection) / np.sum(union)
    print("Jaccard Index: " + str(jaccard))
    return jaccard


def dice_coefficient(ground_truth, prediction, threshold=127):
    ground_truth = moil.getBinaryThreshold(ground_truth, threshold)
    prediction = moil.getBinaryThreshold(prediction, threshold)
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    dice = (2 * np.sum(intersection)) / (np.sum(intersection) + np.sum(union))
    print("Dice Sorensen: " + str(dice))
    return dice
