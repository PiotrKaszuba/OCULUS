import copy
import math

import cv2
import numpy as np


def stackImageChannels(image, original_channel=0, channels=3):
    w, h, c = getWidthHeightChannels(image)
    image = image.reshape(h, w, c)
    array = np.zeros((h, w, channels), dtype=image.dtype)

    for i in range(channels):
        array[:, :, i] = image[:, :, original_channel]

    return array


def morphMultiClosing(image, iterations=0, kernel=None):
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    return eroded


def selectBiggerCircularContour(image, area_growth_ratio=0.2):
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    AreaScalling = 2
    rootDegree = AreaScalling - area_growth_ratio
    powerIndex = 1 / rootDegree

    maxRank = 0.0
    maxRankInd = None

    for i in range(len(contours)):
        try:
            contourArea = cv2.contourArea(contours[i])
            AreaRatio = math.pow(contourArea, powerIndex)
            arcLength = cv2.arcLength(contours[i], True)
            rank = AreaRatio / arcLength
        except:
            rank = 0
        if rank > maxRank:
            maxRank = rank
            maxRankInd = i

    return contours[maxRankInd] if maxRankInd is not None else None


def convertImageType(image, type, maxValue, oldMaxValue):
    if image.dtype == type:
        return image
    image = image / oldMaxValue * maxValue
    return image.astype(type)


def convertToNetRangeFromByte(image):
    return convertImageType(image, np.float32, 1, 255)


def convertToByteRangeFromNet(image):
    return convertImageType(image, np.uint8, 255, 1)


def rollupBeginningVirtualDimensions(image):
    shape = np.shape(image)
    for i in range(len(shape)):
        if shape[i] > 1:
            return image.reshape(shape[i:])


def getBinaryThreshold(image, threshold=127, maxVal=255):
    shape = np.shape(image)
    # loses info about shape of size 1
    ret, thresh = cv2.threshold(image, threshold, maxVal, cv2.THRESH_BINARY)
    return thresh.reshape(shape)


def convertImageNetOutput(image):
    image = rollupBeginningVirtualDimensions(image)
    image = convertToByteRangeFromNet(image)
    return image


def getColsRows(level, base_scale, use_col=True):
    row_multi = 16
    col_multi = 16
    if use_col:
        cols = int(col_multi * level)
        rows = int(round(level * base_scale) * row_multi)
    else:
        rows = int(row_multi * level)
        cols = int(round(level * base_scale) * col_multi)

    return cols, rows


def getWidthHeightChannels(image):
    x = np.shape(image)
    if len(x) == 2:
        return x[1], x[0], 1
    if len(x) == 3:
        return x[1], x[0], x[2]


def show(im, function_on_im=None, *args, other_im=[], function_on_other=None, print=False, window='win'):
    if print:
        print_info(im)

    while 1:
        im_func = None
        if function_on_im is not None:
            im_func = function_on_im(im, *args)
            cv2.imshow(window, im_func)
        else:
            cv2.imshow(window, im)
        temp = 0
        for mat in other_im:
            temp += 1
            if function_on_other is not None:
                mat_func = function_on_other(mat, *args)
                cv2.imshow(window + str(temp), mat_func)
            else:
                cv2.imshow(window + str(temp), mat)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    return im_func


def print_info(im):
    average = np.mean(im)
    h, w = np.shape(im)
    print("pixel average: " + str(average) + ", height: "
          + str(h) + ", width: " + str(w)
          + ", h/w: " + str(h / w)
          )


# modify proportion of height / width of img
# modify_height == True -> modifies height, else modifies width
# not_modified_dim_wanted_val > 0 -> sets the other value to this value
# h_div_w is wanted height/width proportion
def modify_h_div_w(img, h_div_w, modify_height=False, not_modified_dim_wanted_val=0):
    h, w = np.shape(img)

    if modify_height:
        if not_modified_dim_wanted_val > 0:
            w = not_modified_dim_wanted_val
        h = int(h_div_w * w)
    else:
        if not_modified_dim_wanted_val > 0:
            h = not_modified_dim_wanted_val
        w = int(h / h_div_w)
    img = cv2.resize(img, (w, h))
    return img


def addToRegionOfInterest(image, x, y, HalfWidth, HalfHeight, Add):
    w, h, c = getWidthHeightChannels(image)
    shapeLen = len(np.shape(image))
    wAdd, hAdd, cAdd = getWidthHeightChannels(Add)
    if shapeLen == 2:
        template = np.zeros(shape=(h + 2 * HalfHeight, w + 2 * HalfWidth), dtype=image.dtype)
        template[HalfHeight:HalfHeight + h, HalfWidth:HalfWidth + w] = image

        template[y:y + hAdd, x:x + wAdd] = Add

    else:
        template = np.zeros(shape=(h + 2 * HalfHeight, w + 2 * HalfWidth, c), dtype=image.dtype)
        template[HalfHeight:HalfHeight + h, HalfWidth:HalfWidth + w, :] = image

        template[y:y + hAdd, x:x + wAdd, :] = Add

    return template[HalfHeight:HalfHeight + h, HalfWidth:HalfWidth + w]


def getRegionOfInterest(image, x, y, HalfWidth, HalfHeight):
    w, h, c = getWidthHeightChannels(image)
    shapeLen = len(np.shape(image))

    if shapeLen == 2:
        template = np.zeros(shape=(h + 2 * HalfHeight, w + 2 * HalfWidth), dtype=image.dtype)
        template[HalfHeight:HalfHeight + h, HalfWidth:HalfWidth + w] = image
    else:
        template = np.zeros(shape=(h + 2 * HalfHeight, w + 2 * HalfWidth, c), dtype=image.dtype)
        template[HalfHeight:HalfHeight + h, HalfWidth:HalfWidth + w, :] = image

    return template[y:y + 2 * HalfHeight, x:x + 2 * HalfWidth]


# read image of name and extension in path
# mode is for greyscale / rgb
def read_and_size(name, path=None, extension='.jpg', scale=0, mode=0, h_div_w=0,
                  modify_height=False,
                  not_modified_wanted_value=0, target_size=None):
    im = cv2.imread(path + name + extension, mode)
    if target_size is None:
        if scale > 0:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        if h_div_w > 0:
            im = modify_h_div_w(im, h_div_w, modify_height, not_modified_wanted_value)
    else:
        im = cv2.resize(im, target_size)

    return im


def read_and_size_with_copy(name, path=None, extension='.jpg', scale=0, mode=0, h_div_w=0,
                            modify_height=False, not_modified_wanted_value=0, target_size=None):
    im = cv2.imread(path + name + extension, mode)
    if target_size is None:
        if scale > 0:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        if h_div_w > 0:
            im = modify_h_div_w(im, h_div_w, modify_height, not_modified_wanted_value)
    else:
        im = cv2.resize(im, target_size)
    im_copy = copy.deepcopy(im)

    return im, im_copy
