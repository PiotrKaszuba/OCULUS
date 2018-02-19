import cv2
import numpy as np
import copy as cp








def callback(x):
    pass
def show(im):
    while cv2.waitKey(30) != ord('q'):
        #cv2.imshow("win", im)
        pass

cv2.namedWindow("win")
cv2.namedWindow("win2")
cv2.namedWindow("histogram")
cv2.namedWindow("histogram2")
#cv2.namedWindow("canny")
cv2.namedWindow("sum")
cv2.namedWindow("sum2")
cv2.namedWindow("final")
depth=5
max_photo_index = 12
name = 0
name2 = 0
imor = cv2.imread("Images/test/"+str(name)+".jpg",1)
imor=cv2.resize(imor, (350,300))
img = cv2.cvtColor(imor, cv2.COLOR_BGR2GRAY)

imor2 = cv2.imread("Images/test/"+str(name2)+".jpg",1)
imor2=cv2.resize(imor2, (350,300))
img2 = cv2.cvtColor(imor2, cv2.COLOR_BGR2GRAY)

#show(img)
#cv2.createTrackbar("Threshold", "win", 0, 255, callback)

#cv2.createTrackbar("Clip","win",0,200,callback)
#cv2.createTrackbar("Tile","win",0,200,callback)
cv2.createTrackbar("Gaussian Blur Sigma","win", 0, 400,callback)
#cv2.createTrackbar("Area","win",0,150,callback)
cv2.createTrackbar("stosunek","win",0,100,callback)
cv2.createTrackbar("level","win",0,255,callback)
cv2.createTrackbar("Fast","win", 0, 150,callback)
#cv2.createTrackbar("Canny Low","win",0,255,callback)
#cv2.createTrackbar("Canny High","win",0,255,callback)
reuse=cp.deepcopy(img)
reuse2=cp.deepcopy(img2)
i = 0
while 1:
    k = cv2.waitKey(30)
    if k == ord('q'):
        show(reuse)
    if k == ord('e'):
        break
    if k == ord('n'):
        if name2==0:
            name2=max_photo_index
        else:
            name2-=1
        imor2 = cv2.imread("Images/test/" + str(name2) + ".jpg", 1)
        imor2 = cv2.resize(imor2, (350, 300))
        img2 = cv2.cvtColor(imor2, cv2.COLOR_BGR2GRAY)
        reuse2 = cp.deepcopy(img2)

    if k == ord('m'):
        if name2 == max_photo_index:
            name2 = 0
        else:
            name2 += 1
        imor2 = cv2.imread("Images/test/" + str(name2) + ".jpg", 1)
        imor2 = cv2.resize(imor2, (350, 300))
        img2 = cv2.cvtColor(imor2, cv2.COLOR_BGR2GRAY)
        reuse2 = cp.deepcopy(img2)

    if k == ord('z'):
        if name == 0:
            name = max_photo_index
        else:
            name -= 1
        imor = cv2.imread("Images/test/" + str(name) + ".jpg", 1)
        imor = cv2.resize(imor, (350, 300))
        img = cv2.cvtColor(imor, cv2.COLOR_BGR2GRAY)
        reuse = cp.deepcopy(img)

    if k == ord('x'):
        if name == max_photo_index:
            name = 0
        else:
            name += 1
        imor = cv2.imread("Images/test/" + str(name) + ".jpg", 1)
        imor = cv2.resize(imor, (350, 300))
        img = cv2.cvtColor(imor, cv2.COLOR_BGR2GRAY)
        reuse = cp.deepcopy(img)
    #abc = cv2.getTrackbarPos("Tile","win")
    #clahe = cv2.createCLAHE(clipLimit=cv2.getTrackbarPos("Clip","win")*0.05, tileGridSize=(1+abc,1+abc))
    hist = reuse
    hist2 =reuse2
    if np.mean(reuse) < cv2.getTrackbarPos("level","win"):
        hist = cv2.equalizeHist(reuse)
    if np.mean(reuse2) < cv2.getTrackbarPos("level", "win"):
        hist2 = cv2.equalizeHist(reuse2)

    hist=cv2.GaussianBlur(hist,(5,5),0.01*cv2.getTrackbarPos("Gaussian Blur Sigma","win"),0)
    hist2=cv2.GaussianBlur(hist2,(5,5),0.01*cv2.getTrackbarPos("Gaussian Blur Sigma","win"),0)

    im=hist
    im2 = hist2
    #cann = cv2.Canny(im, cv2.getTrackbarPos("Canny Low","win"),cv2.getTrackbarPos("Canny High","win"))



    ims1=cv2.Scharr(im,depth,1,0)
    ims1 = cv2.convertScaleAbs(ims1)

    im2s1 = cv2.Scharr(im2, depth, 1, 0)
    im2s1 = cv2.convertScaleAbs(im2s1)

    ims2=cv2.Scharr(im,depth,0,1)
    ims2 = cv2.convertScaleAbs(ims2)

    im2s2 = cv2.Scharr(im2, depth, 0, 1)
    im2s2 = cv2.convertScaleAbs(im2s2)

    ims3=cv2.Laplacian(im, 5)
    ims3=cv2.convertScaleAbs(ims3)

    im2s3 = cv2.Laplacian(im2, 5)
    im2s3 = cv2.convertScaleAbs(im2s3)



    im = 0.005*cv2.getTrackbarPos("stosunek","win")*ims1+0.005*cv2.getTrackbarPos("stosunek","win")*ims2+(1-0.01*cv2.getTrackbarPos("stosunek","win"))*ims3

    im2 = 0.005*cv2.getTrackbarPos("stosunek","win")*im2s1+0.005*cv2.getTrackbarPos("stosunek","win")*im2s2+(1-0.01*cv2.getTrackbarPos("stosunek","win"))*im2s3


    sum = cv2.convertScaleAbs(im)

    sum2 = cv2.convertScaleAbs(im2)

    fast = cv2.FastFeatureDetector_create(threshold=cv2.getTrackbarPos("Fast","win"), nonmaxSuppression=True)

    points = fast.detect(sum, None)
    points2 = fast.detect(sum2, None)

    orb = cv2.ORB_create()
    points= orb.detect(sum)
    points2 = orb.detect(sum2)
    points, des = orb.compute(sum, points)
    points2, des2 = orb.compute(sum2, points2)

    match = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = match.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if k == ord('s'):
        i+=10
    if k == ord('a'):
        i-=10
    if k == ord('d'):
            i = 0
    final = cv2.drawMatches(sum,points, sum2, points2, matches[i:i+10], None,flags=2)

    #im=cv2.GaussianBlur(im,(5,5),2,0)
    #im = cv2.Canny(im,100,200)

    #im = cv2.equalizeHist(im)
    #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(64,64))
    #im = clahe.apply(im)

    #ims = cp.deepcopy(im)
    #ad=ims
    #while cv2.waitKey(30) != ord('w'):
    #ret, ad = cv2.threshold(ims, cv2.getTrackbarPos("Threshold","win"), 255, cv2.THRESH_BINARY)
    #ad = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, cv2.getTrackbarPos("t1","win")*2+3, 5)
    #ad = cv2.morphologyEx(ad, cv2.MORPH_OPEN, (3, 3))
    #ad = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, (3, 3))

    #ad = cv2.medianBlur(ad, 5)



    #imdraw = cp.deepcopy(imor)
    '''
    imb, contours, hierarchy = cv2.findContours(ad,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(imdraw, contours, -1, (0,255,0), 2)
    for i in range(len(contours) - 1):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        #perimeter = cv2.arcLength(cnt, True)
        if area > cv2.getTrackbarPos("Area","win")-1:
            cv2.drawContours(imdraw, contours, i, (0, 255, 0), 2)
    '''



    cv2.imshow("win", img)
    cv2.imshow("win2", img2)
    cv2.imshow("histogram",hist)
    cv2.imshow("histogram2", hist2)
    #cv2.imshow("canny", cann)
    cv2.imshow("sum", sum)
    cv2.imshow("sum2", sum2)
    cv2.imshow("final", final)
    #show(im)
    #show(im)

