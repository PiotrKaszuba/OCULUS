import cv2
import Code.Libraries.MyOculusImageLib as moil
import Code.Libraries.MyOculusRepoNav as morn
import os
path = '../../../Images/ZanikGray50/'

dict = {}

while True:
    im_path = morn.next_path(path, dict)

    dict[im_path] = True
    if not os.path.exists(im_path):
        continue
    ims = os.listdir(im_path)
    for im in ims:
        if not os.path.isfile(im_path+im):
            continue
        img = moil.read_and_size(name=im, path=im_path, scale=1, extension='')
        img2 = cv2.equalizeHist(img)
        moil.show(img, other_im=[img2])