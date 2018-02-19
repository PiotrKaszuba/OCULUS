import cv2
import numpy as np
import MyOculusLib as mol

depth = 5
mol.init()
im,im2=mol.read_and_size_with_copy(name=str(0), path="Images/test/")



mol.show(im, other_im=[im2])


im = cv2.equalizeHist(im)
im2 = cv2.equalizeHist(im2)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
#im = clahe.apply(im)
#im2 = clahe.apply(im2)
mol.show(im, other_im=[im2])
im2 = cv2.bilateralFilter(im2,3,75,75)
mol.show(im, other_im=[im2])
img1=cv2.Sobel(im,depth,1,0, ksize=3)
img1 = cv2.convertScaleAbs(img1)

im2g1=cv2.Sobel(im2,depth,1,0, ksize=3)
im2g1 = cv2.convertScaleAbs(im2g1)

mol.show(img1, other_im=[im2g1])

img2=cv2.Sobel(im,depth,0,1, ksize=3)
img2 = cv2.convertScaleAbs(img2)

im2g2=cv2.Sobel(im2,depth,0,1, ksize=3)
im2g2 = cv2.convertScaleAbs(im2g2)


img3=cv2.Laplacian(im, 5)
img3=cv2.convertScaleAbs(img3)

im2g3=cv2.Laplacian(im2, 5)
im2g3=cv2.convertScaleAbs(im2g3)



im = 0.5*img1+0.5*img2+0*img3
im2 = 0.5*im2g1+0.5*im2g2+0*im2g3


im = cv2.convertScaleAbs(im)
im2 = cv2.convertScaleAbs(im2)


mol.show(im, other_im=[im2])




