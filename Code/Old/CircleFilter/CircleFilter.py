import cv2
import numpy as np

import Code.Libraries.MyOculusImageLib as moil


###### <    CIRCLE FILTER METHODS  >
# checks if given point x,y is inside the circle of r radius
# circle center a,b is r,r when not given
def in_circle(x, y, r, a=None, b=None):
    if a is None:
        a = r
    if b is None:
        b = r
    if (x - a) ** 2 + (y - b) ** 2 <= r ** 2:
        return True
    else:
        return False


# creates a kernel with given value(default = 1) of given data type in circular area from middle
# with radius of r and zeros in the rest of positions, kernel sizes are 2*r+1 for symmetric and centered
def circle_kernel(r, v=1, type=np.uint8):
    # r must be integer and higher than 0
    assert r > 0 and r % 1 == 0
    kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=type)
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            if in_circle(i, j, r, r, r):
                kernel[i, j] = v
    return kernel


def circle_filter_2d(im, r=None):
    h, w = np.shape(im)
    if r is None:
        r = int(h / 8)
    im = cv2.filter2D(im, 5, circle_kernel(r))
    im /= (r ** 2 * 3.14)
    im = cv2.convertScaleAbs(im)

    return im


def minus(im_subtracted, subtract, negative=False):
    temp = np.int16(im_subtracted) - np.int16(subtract)

    temp[temp < 0] = 0

    temp = np.uint8(temp)
    if negative:
        temp = cv2.bitwise_not(temp)
    return temp


def diff(im_diff, diff):
    temp = np.int16(im_diff) - np.int16(diff)
    temp = np.absolute(temp)
    return np.uint8(temp)


def minus_mean(im_subtracted, im_mean=None, negative=False):
    if im_mean is None:
        im_mean = im_subtracted

    m = np.mean(im_mean)

    return minus(im_subtracted, im_mean, negative)


def mean_diff(im_diff, im_mean=None):
    if im_mean is None:
        im_mean = im_diff

    m = np.mean(im_mean)
    return diff(im_diff, im_mean)


def square_kernel(r, v=1, type=np.uint8):
    # r must be integer and higher than 0
    assert r > 0 and r % 1 == 0
    kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=type)
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            kernel[i, j] = v
    return kernel


def square_filter_2d(im, r=None):
    h, w = np.shape(im)
    if r is None:
        r = (int)(h / 8)
    im = cv2.filter2D(im, 5, square_kernel(r))
    im /= (r ** 2 * 4)
    im = cv2.convertScaleAbs(im)

    return im


def square_circle_difference_filter_2d(im, r=None):
    return diff(square_filter_2d(im, r), circle_filter_2d(im, r))


def square_circle_minus_filter_2d(im, r=None, negative=False):
    if negative:
        return minus(square_filter_2d(im, r), circle_filter_2d(im, r))
    else:
        return minus(circle_filter_2d(im, r), square_filter_2d(im, r))


###### <    CIRCLE FILTER METHODS />

def equalize_border_with_mean_or_val(im, val=None, border_base=0, border_max_offset=None):
    average = np.mean(im)

    if border_max_offset is None:
        border_max_offset = int((average / 10) - 1)

    x, y = (abs(im - border_base) < border_max_offset).nonzero()

    if val is None:
        size = np.size(im)
        l = len(x)
        val = (average * size) / (size - l)

    im[x, y] = val
    return im


def square_circle_on_1_2_in_path(path):
    for i in range(6):
        im, im_cp = moil.read_and_size_with_copy(str(i), path=path, scale=0.2)

        im = equalize_border_with_mean_or_val(im)

        h, w = np.shape(im)
        r = (int)(h / 8)
        img_sqd = square_circle_difference_filter_2d(im, r)
        # img_sqd = square_circle_minus_filter_2d(im, r)
        # img_sqd = circle_filter_2d(im)

        img_sqd = cv2.equalizeHist(img_sqd)

        ret, imt = cv2.threshold(img_sqd, 250, 255, cv2.THRESH_BINARY)
        cv2.imshow('win2', im_cp)
        cv2.imshow('win32', img_sqd)
        moil.show(imt)

        '''imt2,cnt, hier = cv2.findContours(imt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w = np.shape(im)
        r = (int)(h / 10)
        for j in range(len(cnt)):
            temp = cnt[j]
            M=cv2.moments(temp)
            cx = int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])

            cv2.circle(im_cp, (cx,cy), r)
            show(im_cp)'''
