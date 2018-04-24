import cv2
import numpy as np
import copy as cp
import random

global im
global a
a=0
def cursor_step(x,y):
    global im
    global a
    if a == 2:
       a=0
    if a==0:
        x+=1
    if a==1:
        y+=1
    if a==2:
        x-=1
    if a==3:
        y-=1
    return x,y




def start_cursor(x,y):
    global a
    while im[y,x]!=127:
        old_x,old_y=x,y
        x,y=cursor_step(x,y)
        if x>= len(im[0]) or x<=0 or y>=len(im) or y<=0:
            x,y=10,10

        if im[y, x]  != im[old_y, old_x] and im[y,x]!=127:
            im[y, x] = 127
            x,y=old_x,old_y
            y+=1
        if x>= len(im[0]) or x<=0 or y>=len(im) or y<=0:
            x,y=10,10





def callback(x):
    pass
def show(im):
    while cv2.waitKey(30) != ord('q'):
        cv2.imshow("win", im)

cv2.namedWindow("win")
depth = 5
thresh = 60
name = "2"


kernel = np.array(([1,0,1],[0,1,0],[1,0,1]), dtype=np.uint8)
kernel2 = np.array(([0,1,0],[1,1,1],[0,1,0]), dtype=np.uint8)
im = cv2.imread("../../Images/test/"+name+".jpg",0)

im=cv2.resize(im, (0,0), fx=0.3, fy=0.3)

show(im)

im = cv2.equalizeHist(im)
#clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(64,64))
#im = clahe.apply(im)


ims=cp.deepcopy(im)
#ims = cv2.cvtColor(ims,cv2.COLOR_GRAY2BGR)
'''
show(im)
#im=cv2.blur(im, (3,3))
im = cv2.bilateralFilter(im,5,75,40)
show(im)
#im = cv2.medianBlur(im, 3)
#im=cv2.erode(im,(3,3), iterations=1)
im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,0)
show(im)
'''
'''
im=cv2.dilate(im,kernel, iterations=1)
show(im)
im=cv2.erode(im,kernel, iterations=1)

show(im)

im=cv2.dilate(im,kernel2, iterations=1)
show(im)
im=cv2.erode(im,kernel2, iterations=1)

#im=cv2.Laplacian(im, 5)
#im = cv2.convertScaleAbs(im)
#show(im)

show(im)
'''

'''
for i in range(len(contours)-1):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area>100.0 :
        cv2.drawContours(ims, contours, i, (0,255,0), cv2.FILLED)
'''

#show(ims)
'''
for i in range(10000):
    y=random.randint(10,len(im)-10)
    x=random.randint(10,len(im[0])-10)

    start_cursor(x,y)


im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)

for a in range(len(im)):
    for b in range (len(im[0])):
        if im[a,b,0] == 127:
            im[a,b,0] = 0
            im[a, b, 1] = 0

#im=cv2.dilate(im,(3,3), iterations=1)
#show(im)
#im=cv2.dilate(im,kernel, iterations=1)
show(im)
#im=cv2.erode(im,kernel, iterations=1)

show(im)





#im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, (5, 5))
'''
show(im)
ksize=3
im1=cv2.Sobel(im,depth,1,0, ksize=ksize)
im1 = cv2.convertScaleAbs(im1)
show(im1)

im2=cv2.Sobel(im,depth,0,1, ksize=ksize)
im2 = cv2.convertScaleAbs(im2)
show(im2)

im3=cv2.Laplacian(im, depth,ksize=ksize)
im3=cv2.convertScaleAbs(im3)
show(im3)


im = 0.4*im1+0.4*im2+0.2*im3

show(im)

im = cv2.convertScaleAbs(im)




show(im)
#im=cv2.GaussianBlur(im,(5,5),2,0)
#im = cv2.Canny(im,100,200)



show(im)
im = cv2.convertScaleAbs(im)
im=ims-im

show(im)

im = cv2.medianBlur(im, 3)

show(im)
#im = cv2.bilateralFilter(im,9,75,75)
im=cv2.blur(im, (3,3))

show(im)
'''











circles = cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,1,10,param1=250,param2=70,minRadius=25,maxRadius=100)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)

'''
#im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#im=cv2.Laplacian(im, depth)



#im = cv2.blur(im, (5,5))





#im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,(3,3))

#im=cv2.dilate(im,(3,3), iterations=3)
#im=cv2.erode(im,(3,3), iterations=3)
##im=cv2.Sobel(im,depth,0,1, ksize=3)
ad = 0
'''
ad = cv2.medianBlur(ad, 3)
ad = cv2.erode(ad, (3, 3), iterations=5)
ad = cv2.dilate(ad, (3, 3), iterations=5)
'''
#ims = cv2.cvtColor(ims,cv2.COLOR_BGR2GRAY)
cv2.createTrackbar("t1", "win", 0, 255, callback)
while cv2.waitKey(30) != ord('q'):
    ret, ad = cv2.threshold(ims, cv2.getTrackbarPos("t1","win"), 255, cv2.THRESH_BINARY)
    #ad = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, cv2.getTrackbarPos("t1","win")*2+3, 5)
    ad = cv2.morphologyEx(ad, cv2.MORPH_OPEN, (3, 3))
    ad = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, (3, 3))

    ad = cv2.medianBlur(ad, 5)
    cv2.imshow("win", ad)

ims=ad
ims = cv2.cvtColor(ims,cv2.COLOR_GRAY2BGR)
im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)-1):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area>300.0 :
        cv2.drawContours(ims, contours, i, (0,255,0), cv2.FILLED)

show(ims)
