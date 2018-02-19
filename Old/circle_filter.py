import cv2
import numpy as np
import math

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

#list of coords should be given
#kernel shapes should be odd
#if coords to center are not given or are less than zero they are centered in material(if possible)
def ret_sum_and_kernel_of_bitwise_and_centered_at_point(material, kernel, coords=None):

    #assert data types
    assert material.dtype == kernel.dtype
    #getting dimensions
    dimension_kernel = np.shape(np.shape(kernel))[0]
    dimension = np.shape(np.shape(material))[0]
    #asserting dimensions are equal, in the implemented range
    assert dimension_kernel == dimension and dimension > 0 and dimension < 3




    #if there are missing coords or invalid (less than 0) place coord in center
    coord_to_generate = np.zeros(dimension, dtype=np.bool)
    #if no coords are given we create new list with coords and mark all to be generated
    if coords == None:
        coords=[0] * dimension
        for i in range(dimension):
            coord_to_generate[i]=1
    else:
        #if some coords are given we append values to be set and mark some to be generated
        if len(coords) < dimension:
            for i in range(dimension - len(coords)):
                coords.append(0)
                coord_to_generate[i+len(coords)]=1

    #finally we check if there are invalid coords overall (less than 0) and set them to be generated
    for i in range(dimension):
        if(coords[i] < 0 ):
            coord_to_generate[i]=1



    #material shapes
    s=[]
    #kernel shapes
    r=[]
    #kernel shapes minus center
    rmc=[]
    #kernel halves of shapes minus center
    hrmc=[]
    for i in range(dimension):
        #getting all dimension values
        s.append(np.shape(material)[i])
        r.append(np.shape(kernel)[i])
        rmc.append(r[i]-1)
        hrmc.append(rmc[i]/2)

        #generating coords
        if coord_to_generate[i] == 1:
            coords[i] = math.floor((s[i]-1)/2)


        #asserting if we can center point in material and kernel wont go outside
        assert rmc[i]%2 == 0 and hrmc[i] > 0
        assert coords[i] - hrmc[i] >= 0 and coords[i] + hrmc[i] <= s[i]

    #output values
    sum = 0
    new_kernel=None

    #for all dimensions do analogical operations
    if dimension == 1:
        #initialize out kernel with proper shapes
        new_kernel = np.zeros((r[0]), dtype=kernel.dtype)

        #loops through all dimensions
        for i in range(r[0]):
            #get new kernel value as a bitwise and of kernel and material centered at coords
            new_kernel[i] = kernel[i] & material[(int)(coords[0] - hrmc[0]) + i]
            #add new kernel value to sum
            sum+=new_kernel[i]


    if dimension == 2:
        new_kernel = np.zeros((r[0],r[1]), dtype=kernel.dtype)

        for i in range(r[0]):
            for j in range(r[1]):
                new_kernel[i,j] = kernel[i,j] & material[(int)(coords[0] - hrmc[0]) + i, (int)(coords[1] - hrmc[1]) + j]
                sum += new_kernel[i,j]

    return sum, new_kernel



def circle_filter(img, r, step=1):

    # r must be integer and higher than zero
    assert r > 0 and r % 1 == 0
    #step must be integer and higher than zero
    assert step > 0 and step%1 == 0
    #dimension of image must be 2
    assert np.shape(np.shape(img))[0] == 2

    h,w = np.shape(img)

    #there must be atleast one out point
    assert h>= 2*r+1 and w >= 2*r+1

    kernel = circle_kernel(r, 255, img.dtype)
    out_h = (int) ((h-2*r-1)/step) +1
    out_w = (int) ((w-2*r-1)/step) +1

    out_img = np.zeros((out_h,out_w),img.dtype)
    for i in range(out_h):
        for j in range(out_w):
            out_img[i,j] = (int)(ret_sum_and_kernel_of_bitwise_and_centered_at_point(img, kernel, [r+i*step,r+j*step])[0]/(3.14*(r**2)))

    return out_img





r=15
step=5
im = cv2.imread('Images/2.jpg',0)
im=cv2.resize(im, (400,400))
im2=cv2.resize(im, (0,0), fx=1/step, fy=1/step)

img = circle_filter(im, r, step)




kern = circle_kernel(r)
img2 = cv2.filter2D(im,5, kern)
img2 /=(r**2 * 3.14)
img2 =cv2.convertScaleAbs(img2)

img2=cv2.resize(img2, (0,0), fx=1/step, fy=1/step)
while(cv2.waitKey(30) != ord('q')):
    cv2.imshow('org',im)
    cv2.imshow('win', img)
    cv2.imshow('win2', im2)
    cv2.imshow('win3', img2)
