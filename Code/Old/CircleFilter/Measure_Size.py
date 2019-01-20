import cv2
import numpy as np
import os
import Code.Old.CircleFilter.CircleFilter as cf
import Code.Libraries.MyOculusImageLib as moil

path = "../../Images/all/"


def get_number_of_images_in_path(image_path):
    return len(i for i in os.listdir(image_path) if os.path.isfile(i))


def square_circle_on_1_2_in_path(path):
    for i in range(2):
        im, im_cp = moil.read_and_size_with_copy(str(i), scale=0.15)

        im = cf.equalize_border_with_mean_or_val(im)

        img_sqd = cf.square_circle_minus_filter_2d(im)

        img_sqd = cv2.equalizeHist(img_sqd)

        ret, imt = cv2.threshold(img_sqd, 254, 255, cv2.THRESH_BINARY)

        imt2, cnt, hier = cv2.findContours(imt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w = np.shape(im)
        r = (int)(h / 10)
        for j in range(len(cnt)):
            temp = cnt[j]
            M = cv2.moments(temp)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(im_cp, (cx, cy), r)

def trackback_callback(x):
    global track_val
    track_val = x

def track(name, max, start =0, win='win'):
    cv2.createTrackbar(name, win, start, max, trackback_callback)

def get_track(name, win='win'):
    return cv2.getTrackbarPos(name, win)

for i in range(get_number_of_images_in_path(path)):

    im, im_cp = moil.read_and_size_with_copy(str(i), scale=0.15)

    cf.equalize_border_with_mean_or_val(im)

    negative = False
    if i >= 2 and i <= 5:
        negative = True

    track('thresh', 255)
    track('r', 100)

    while (cv2.waitKey(30) != ord('q')):
        img = cf.circle_filter_2d(im, get_track('r') + 1)
        img_sqd = cf.square_circle_minus_filter_2d(im, get_track('r') + 1, negative)

        img_sqd = cv2.equalizeHist(img_sqd)
        img = cv2.equalizeHist(img)

        ret, imt = cv2.threshold(img_sqd, get_track('thresh'), 255, cv2.THRESH_BINARY)
        ret2, imt2 = cv2.threshold(img, get_track('thresh'), 255, cv2.THRESH_BINARY)

        cv2.imshow('win', im)
        cv2.imshow('circle', imt2)
        cv2.imshow('square', imt)
        cv2.imshow('cir', img)
        cv2.imshow('sq', img_sqd)
