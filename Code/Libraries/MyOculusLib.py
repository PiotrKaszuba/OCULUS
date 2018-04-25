import cv2
import numpy as np
import math
import copy
import os
import shutil
import csv
from functools import reduce

####### <   Globals >
masks_done =0
patients_done=0
patient_inds = 0
patient_date = 0
eye = 'left'
winname = 'win'
measure_size_clicks_ptr = []
measure_size_clicks = 0
image_path =''
track_val =0
height=0
width=0
mask = None
accepted = True
rr = 0
xx = 0
yy = 0

###### <    Globals />



###### <    CIRCLE FILTER METHODS  >
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


def circle_filter_2d(im, r=None):
    h,w = np.shape(im)
    if r==None:
        r=(int)(h/8)
    im = cv2.filter2D(im, 5, circle_kernel(r))
    im /= (r ** 2 * 3.14)
    im = cv2.convertScaleAbs(im)

    return im

def minus(im_subtracted, subtract, negative=False):
    temp = np.int16(im_subtracted) - np.int16(subtract)

    temp[temp < 0] = 0

    temp =np.uint8(temp)
    if negative:
        temp = cv2.bitwise_not(temp)
    return temp

def diff(im_diff, diff):
    temp = np.int16(im_diff) - np.int16(diff)
    temp = np.absolute(temp)
    return np.uint8(temp)

def minus_mean(im_subtracted, im_mean=None, negative=False):
    if im_mean == None:
        im_mean=im_subtracted

    m = np.mean(im_mean)

    return minus(im_subtracted, im_mean, negative)

def mean_diff(im_diff, im_mean=None):
    if im_mean == None:
        im_mean=im_diff

    m = np.mean(im_mean)
    return diff(im_diff,im_mean)


def square_kernel(r, v=1, type=np.uint8):
    # r must be integer and higher than 0
    assert r > 0 and r % 1 == 0
    kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=type)
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            kernel[i, j] = v
    return kernel

def square_filter_2d(im, r=None):
    h,w = np.shape(im)
    if r==None:
        r=(int)(h/8)
    im = cv2.filter2D(im, 5, square_kernel(r))
    im /= (r ** 2 * 4)
    im = cv2.convertScaleAbs(im)

    return im


def square_circle_difference_filter_2d(im, r=None):
    return diff(square_filter_2d(im,r),circle_filter_2d(im,r))

def square_circle_minus_filter_2d(im, r=None, negative=False):
    if negative:
        return minus(square_filter_2d(im, r), circle_filter_2d(im, r))
    else:
        return minus(circle_filter_2d(im,r), square_filter_2d(im,r))
###### <    CIRCLE FILTER METHODS />



###### <     HELP METHODS    >


def print_info(im):
    average = np.mean(im)
    h, w = np.shape(im)
    print("pixel average: " + str(average) + ", height: "
          + str(h) + ", width: " + str(w)
          + ", h/w: " + str(h/w)

          )


def all_path(func,start_path=None, eye=None):
    global patients_done
    if eye != 'left' and eye != 'right':
        eye='both'

    patient = None
    if start_path==None:
        start_path=image_path


    patient = os.listdir(start_path)
    for i in range (len([a for a in patient if not os.path.isfile(a)])):
        date = os.listdir(start_path + patient[i] + "/")

        for j in range(len(date)):
            if eye == 'left':
                func(start_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/')
            if eye == 'right':
                func(start_path + patient[i] + '/' + date[j] + '/' + 'right_eye_images/')
            if eye == 'both':
                func(start_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/')
                func(start_path + patient[i] + '/' + date[j] + '/' + 'right_eye_images/')
        patients_done+=1
        print("patients: " +str(patients_done))
def random_path(start_path=None, eye = None):
    if eye != 'left' and eye != 'right':
        if np.random.randint(0,2) ==0:
            eye='left'
        else:
            eye='right'


    patient = None
    global image_path
    if start_path==None:
        start_path=image_path

    patient = os.listdir(start_path)

    leng = len([f for f in patient if not os.path.isfile(os.path.join(start_path, f))])

    patient_path = patient[np.random.randint(0,leng)]

    date = os.listdir(start_path+patient_path+"/")

    date_path = date[np.random.randint(0,len(date))]

    if start_path=='':
        return patient_path+"/"+date_path+"/"+eye+"_eye_images/"
    else:
        return start_path+"/"+patient_path + "/" + date_path + "/" + eye + "_eye_images/"

def get_number_of_images_in_path():
    global image_path
    return len(i for i in os.listdir(image_path) if os.path.isfile(i))


def modify_h_div_w(img, h_div_w, modify_height=False, not_modified_dim_wanted_val=0):


    h,w = np.shape(img)


    if modify_height:
        if not_modified_dim_wanted_val > 0:
            w = not_modified_dim_wanted_val
        h = (int)(h_div_w*w)
    else:
        if not_modified_dim_wanted_val > 0:
            h = not_modified_dim_wanted_val
        w = (int) (h/h_div_w)
    img = cv2.resize(img, (w,h))
    return img

def read_and_size( name, path=None, extension='.jpg', scale=0, mode=0, modify_shape=True, h_div_w = 0, modify_height=False, not_modified_wanted_value=0, target_size=None):
    global image_path
    if path == None:
        path = image_path
    im = cv2.imread(path+name+extension, mode)
    if target_size ==None:
        if scale >0 :
            im = cv2.resize(im, (0,0), fx=scale, fy=scale)
        if h_div_w>0:
            im = modify_h_div_w(im,h_div_w,modify_height,not_modified_wanted_value)
    else:
        im = cv2.resize(im, target_size)
    if modify_shape:
        load_h_w_from_img(im)
    return im

def read_and_size_with_copy( name, path=None, extension='.jpg', scale=0, mode=0, modify_shape=True, h_div_w = 0, modify_height=False, not_modified_wanted_value=0, target_size=None):
    global image_path
    if path == None:
        path = image_path
    im = cv2.imread(path+name+extension, mode)
    if target_size == None:
        if scale > 0:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        if h_div_w > 0:
            im = modify_h_div_w(im, h_div_w, modify_height, not_modified_wanted_value)
    else:
        im = cv2.resize(im, target_size)
    im_copy = copy.deepcopy(im)
    if modify_shape:
        load_h_w_from_img(im)
    return im, im_copy

def trackback_callback(x):
    global track_val
    track_val = x

def show(im, function_on_im=None, *args, other_im = [], function_on_other=None, print=False, window=None):
    global winname
    if window==None:
        window = winname
    key = None
    if print:
        print_info(im)

    while 1:
        im_func = None
        if function_on_im != None:
            im_func = function_on_im(im, *args)
            cv2.imshow(window, im_func)
        else:
            cv2.imshow(window, im)
        temp = 0
        for mat in other_im:
            temp += 1
            if function_on_other != None:
                mat_func = function_on_other(mat, *args)
                cv2.imshow(window + str(temp), mat_func)
            else:
                cv2.imshow(window+str(temp), mat)
        key =cv2.waitKey(30)
        if key == ord('q'):
            break
    return im_func

def track(name, max, start =0, win='win'):
    global winname
    if winname != win and win=='win':
        win = winname
    cv2.createTrackbar(name, win, start, max, trackback_callback)

def get_track(name, win='win'):
    global winname
    if winname != win and win == 'win':
        win = winname
    return cv2.getTrackbarPos(name, win)


###### <     HELP METHODS    />



###### < MOUSE MEASURE SIZE METHODS >


def measure_size(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global measure_size_clicks, measure_size_clicks_ptr
        measure_size_clicks+=1
        measure_size_clicks_ptr.append((x, y))
        if measure_size_clicks == 2:
            print('x difference: ' + str(abs(measure_size_clicks_ptr[0][0] - measure_size_clicks_ptr[1][0])))
            print('y difference: ' + str(abs(measure_size_clicks_ptr[0][1] - measure_size_clicks_ptr[1][1])))
            measure_size_clicks_ptr = []
            measure_size_clicks = 0
    if event == cv2.EVENT_RBUTTONDOWN:
        print('point: ' + str(x) +', ' + str(y))


def draw_circle(event, x, y, flags, param):
    global width, height, mask, rr, xx, yy, accepted

    if event == cv2.EVENT_LBUTTONDOWN:
        accepted = False
        xx = x
        yy = y
        mask = np.zeros((height,width), dtype=np.uint8)
        mask =cv2.circle(mask,(xx,yy), rr, 255, -1)
        cv2.imshow('mask', mask)

    if event == cv2.EVENT_RBUTTONDOWN:

        accepted = False
        cv2.destroyWindow('a')
        cv2.destroyWindow('mask')

    if event == 10:
        accepted = False
        if (flags>0):
            rr+=1

        else:
            rr-=1
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = cv2.circle(mask, (xx, yy), rr, 255, -1)
        cv2.imshow('mask', mask)


def draw_elipse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global measure_size_clicks, measure_size_clicks_ptr,width,height,mask
        measure_size_clicks+=1
        measure_size_clicks_ptr.append([x, y])
        if measure_size_clicks == 4:


            xmax=  np.max(measure_size_clicks_ptr, axis=0)[0]
            ymax= np.max(measure_size_clicks_ptr, axis=0)[1]

            xmin=np.min(measure_size_clicks_ptr, axis=0)[0]
            ymin=np.min(measure_size_clicks_ptr, axis=0)[1]

            orientation_point_ind = np.argmin(measure_size_clicks_ptr,axis=0)[1]

            orientation_point= measure_size_clicks_ptr[orientation_point_ind]


            xcenter = (int)((xmax+xmin)/2)
            ycenter = (int)((ymax+ymin)/2)


            try:
                deltax=orientation_point[0]-xcenter
                deltay = orientation_point[1] - ycenter

                vector = deltax/deltay
                rad =math.atan(vector)

                deg =math.degrees(rad)



            except ZeroDivisionError:
                deg=0.0
            #deg=0.0

            xaxe = (int)((xmax - xmin) / 2)
            yaxe = (int)((ymax - ymin) / 2)
            if xaxe > yaxe:
                deg = -deg
            mask = np.zeros((height,width), dtype=np.uint8)

            mask =cv2.ellipse(mask ,(xcenter,ycenter), (xaxe,yaxe), -deg, 0.0,360.0,255,-1)
            cv2.imshow('mask', mask)
            measure_size_clicks=0
            measure_size_clicks_ptr=[]
    if event == cv2.EVENT_RBUTTONDOWN:
        global accepted
        accepted = False
        cv2.destroyWindow('a')
        cv2.destroyWindow('mask')
###### < MOUSE MEASURE SIZE METHODS />




# < General processing >

#takes mean if val is None
def equalize_border_with_mean_or_val(im, val = None, border_base=0, border_max_offset=None):


    average = np.mean(im)

    if border_max_offset == None:
        border_max_offset = (int)((average / 10) - 1)

    x, y = (abs(im-border_base) < border_max_offset).nonzero()

    if val == None:
        size = np.size(im)
        l = len(x)
        val = (average * size) / (size - l)


    im[x, y] = val
    return im


# < General processing />



# <Init>
def set_h_w(h, w):
    global height
    global width
    height = h
    width = w

def load_h_w_from_img(img):
    x = np.shape(img)
    global height
    global width
    height = x[0]
    width = x[1]

def init(win='win', im_path = 'Images/all/', start_path='Images/all/', mouse_f=draw_circle, w=518, h=392):
    global winname
    global image_path
    global height
    global width
    height=h
    width=w


    cv2.destroyWindow(winname)
    if im_path==None:
        image_path=random_path(start_path)
    else:
        image_path=im_path
    winname = win
    measure_size_clicks_ptr = []
    measure_size_clicks = 0
    cv2.namedWindow(winname)

    cv2.setMouseCallback(winname, mouse_f)

init(im_path='')

#<Init/>

def delete_directories(lines_of_directories, path):
    list = lines_of_directories.split('\n')
    for item in list:

        shutil.rmtree(path+item)


#returns same patient lines in filename and local directory
def same_lines(filename, path_to_txt, path_to_local):
    string_local = string_patients(path_to_local)
    file = open(path_to_txt+filename+'.txt', 'r')
    string_file = file.read()

    local_list = string_local.split('\n')
    file_list = sorted(string_file.split('\n'))
    string = ''
    i=0
    imax = len(local_list)
    j=0
    jmax = len(file_list)

    while(i!=imax and j!=jmax):
        if(local_list[i] > file_list[j]):
            j+=1
        if (local_list[i] < file_list[j]):
            i += 1
        if (local_list[i] == file_list[j]):
            string += local_list[i]+'\n'
            i+=1
            j+=1

    return string


def to_txt(string,name, path_to_txt):
    file = open(path_to_txt+name+'.txt','w')
    file.write(string)


#lists all patients in local directory
def string_patients(path):
    list = os.listdir(path)
    string = ''
    for item in sorted(list):
        string+= item+'\n'
    return string


def random_image_on_path(path, target_size=None, ret_numb=False):
    numb = len(i for i in os.listdir(path) if os.path.isfile(i))
    j = np.random.randint(numb)
    im=read_and_size(str(j), path=path, target_size=target_size)
    if ret_numb:
        return im, j
    return im


def createImagesRepo(base_repo_path, repo_name):
    if not os.path.exists(base_repo_path+repo_name):
        os.makedirs(base_repo_path+repo_name)
        return True
    return False

def createImageInPath(path, name, image, override=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path+name):
        cv2.imwrite(path+name, image)
        return True
    if override:
        cv2.imwrite(path + name, image)
        return True
    return False




def func_on_random_images(func, target_size, times=1, start_path=None, eye=None):
    for i in range(times):
        path = random_path(start_path,eye)
        img = random_image_on_path(path, target_size)
        func(img)

def getRepoPathAndImagePath(path):
    path = path.split('/')[:-1]
    repo = reduce((lambda x, y: x + '/' + y), path[:len(path) - 3])+'/'
    image = reduce((lambda x, y: x + '/' + y), path[-3:])+'/'
    return repo, image
def getBaseRepoPathAndRepoName(repo_path):
    repo_path = repo_path.split('/')[:-1]
    base = reduce((lambda x, y: x + '/' + y), repo_path[:len(repo_path) - 1])
    name = repo_path[-1]
    return base, name

def getPatientDateEye(image_path):
    image_path=image_path.split('/')[:-1]
    return image_path[0], image_path[1], image_path[2]

def createFromAllPathImageAfterFunction(old_repo_path, new_repo_path, function, target_size=None, eye=None, override=False, extension='.jpg'):
    success = 0
    fail = 0
    if eye != 'left' and eye != 'right':
        eye='both'

    patient = None


    patient = os.listdir(old_repo_path)
    for i in range (len([a for a in patient if not os.path.isfile(os.path.join(old_repo_path,a))])):
        date = os.listdir(old_repo_path + patient[i] + "/")

        for j in range(len(date)):
            if eye == 'left':
                t1, t2 = createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/', new_repo_path,function,target_size,override, extension)
                success+=t1
                fail+=t2
            if eye == 'right':
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'right_eye_images/', new_repo_path,function,target_size,override, extension)
                success += t1
                fail += t2
            if eye == 'both':
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/', new_repo_path,function,target_size,override, extension)
                success += t1
                fail += t2
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'right_eye_images/', new_repo_path,function,target_size,override, extension)
                success += t1
                fail += t2
    print("Images created: " + str(success) + ", attempts failed to create: " + str(fail))
def createFromRandomImageAfterFunction(old_repo_path, new_repo_path, function, target_size=None, times=1, eye=None, override=False, extension='.jpg'):
    success = 0
    fail = 0
    for i in range(times):
        rand_path = random_path(old_repo_path, eye)
        img, numb = random_image_on_path(rand_path,target_size,ret_numb=True)

        repo_path, image_path=getRepoPathAndImagePath(rand_path)
        new_path = new_repo_path+image_path

        if createImageInRepoAfterFunction(new_path,str(numb)+extension,img, function,override):
            success+=1
        else:
            fail+=1
    print("Images created: " + str(success) + ", attempts failed to create: " + str(fail))



def createImagesInRepoAfterFunctionOnPath(path, new_repo_path, function, target_size, override=False, extension='.jpg'):
    success = 0
    fail = 0
    repo_path, image_path = getRepoPathAndImagePath(path)

    base2, name2 = getBaseRepoPathAndRepoName(new_repo_path)
    createImagesRepo(base2, name2)
    new_path = new_repo_path+image_path


    for a in os.listdir(path):
        if not os.path.isfile(os.path.join(path, a)):
            continue
        name = a.split(".")[0]
        base_image = read_and_size(name, path=path, target_size=target_size)
        image = function(base_image)

        if createImageInPath(new_path, name+extension, image, override):
            registerImageCsv(new_repo_path,image_path,name+extension,image,function)
            success+=1
        else:
            fail+=1
    return success, fail
def createImageInRepoAfterFunction(path, image_name, base_image, function, override=False):
    repo_path, image_path = getRepoPathAndImagePath(path)
    base, name = getBaseRepoPathAndRepoName(repo_path)
    createImagesRepo(base,name)
    image = function(base_image)
    if createImageInPath(repo_path+image_path, image_name, image, override):
        registerImageCsv(repo_path,image_path,image_name,image,function)
        return True
    else:
        return False

def getWidthHeightChannels(image):
    x = np.shape(image)
    if len(x) == 2:
        return x[1], x[0], 1
    if len(x) == 3:
        return x[1], x[0], x[2]

def writeToCsv(path, header, row):

    if not os.path.isfile(path):
        csvFile = open(path, 'w', newline="")
        writer = csv.writer(csvFile)
        writer.writerow(header)
        csvFile.close()

    csvFile = open(path, 'a', newline="")
    writer = csv.writer(csvFile)
    writer.writerow(row)
    csvFile.close()

def registerImageCsv(repo_path, image_path, image_name, image, function):
    patient, date, eye = getPatientDateEye(image_path)
    width, height, channels = getWidthHeightChannels(image)
    func_name = function.__name__
    header = ['patient', 'date', 'eye', 'name', 'width', 'height', 'channels', 'function']
    row = [patient, date, eye, image_name, width, height, channels, func_name]
    writeToCsv(repo_path+"imageData.csv", header, row)


def getCsvList(repo_path, image=True):

    list = []
    if image:
        name = "imageData.csv"
    else:
        name= "maskData.csv"
    with open(repo_path + name, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            list.append(row)
    return list
def createMaskFromCsv(repo_path, imageRow, list = None, override=False):
    if list is None:
        list = getCsvList(repo_path, image=False)

    target = None

    for row in list:
        equal = True
        for j in range(4):
            if row[j] != imageRow[j]:
                equal=False
                break
        if equal:
            target = row
            break
    if target is None:
        return False

    imageW = int(imageRow[4])
    imageH = int(imageRow[5])

    maskW = int(target[4])
    maskH = int(target[5])
    maskX = int(target[6])
    maskY = int(target[7])
    maskR = int(target[8])

    Wratio = imageW/maskW
    Hratio = imageH / maskH

    Rratio = math.sqrt((((Wratio**2)+(Hratio**2))/2))

    outX = int(maskX*(Wratio))
    outY = int(maskY * (Hratio))
    outR = int(maskR * (Rratio))
    mask = np.zeros((imageH, imageW), dtype=np.uint8)
    mask = cv2.circle(mask, (outX, outY), outR, 255, -1)

    path = repo_path + reduce((lambda x, y: x + '/' + y), imageRow[:3])+'/mask/'

    return createImageInPath(path,imageRow[3],mask,override)


def checkIfPresentInCsv(repo_path, imageRow, list=None, image=True):
    if list is None:
        list= getCsvList(repo_path, image)
    for row in list:
        equal = True
        for j in range(4):
            if row[j] != imageRow[j]:
                equal=False
                break
        if equal:
            if image:
                print("There is already image "+ reduce((lambda x, y: x + '/' + y), imageRow[:4]))
            else:
                print("There is already mask " + reduce((lambda x, y: x + '/' + y), imageRow[:3]) + '/mask/'+imageRow[3])
            return False
    return True


def createAllMasksForImagesCsv(repo_path):
    success = 0
    fail = 0
    list = getCsvList(repo_path, image=False)

    with open(repo_path + "imageData.csv", 'r') as file:
        reader = csv.reader(file)
        next(reader,None)
        for row in reader:
            if createMaskFromCsv(repo_path,row,list,True):
                success+=1
            else:
                fail+=1
        file.close()
    print("Masks created: "+str(success)+ ", failed to create: " +str(fail))


def circle_mask_on_path(path, target_size=None, r = None,extension=".jpg"):
    global accepted, winname, masks_done, rr
    if r == None and target_size !=None:
        rr = int(target_size[0]/10)
    else:
        rr = r
    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    for i in range( len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])):
        if os.path.exists(path+'/mask/'+str(i)+'.jpg'):
            masks_done+=1
            continue
        ImName = str(i)+extension
        img = read_and_size(str(i), path=path, target_size=target_size, extension=extension)
        w,h,c = getWidthHeightChannels(img)
        if r == None and target_size == None:
            rr = int(w / 10)
            target_size = (w, h)
        show(img)

        accepted = False
        while (not accepted):
            accepted = True

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im2 = copy.deepcopy(img)
            cv2.drawContours(im2, contours, 0, (0, 255, 255), 2)

            show(im2)

        split_path = path.split("/")[:-1]

        repo_path = reduce((lambda x, y: x + '/' + y), split_path[:len(split_path) - 3])
        if not os.path.isfile(repo_path + "/maskData.csv"):
            csvFile = open(repo_path + '/maskData.csv', 'w', newline="")
            writer = csv.writer(csvFile)
            writer.writerow(['patient', 'date', 'eye', 'name', 'width', 'height', 'x', 'y', 'r'])
            csvFile.close()

        csvFile = open(repo_path + '/maskData.csv', 'a', newline="")
        writer = csv.writer(csvFile)
        ls = split_path[-3:]
        ls.extend([ImName, target_size[0], target_size[1], xx, yy, rr])
        writer.writerow(ls)
        csvFile.close()
        cv2.imwrite(path + '/mask/' + ImName, mask)
        masks_done += 1
        print("masks: " + str(masks_done))
        cv2.destroyWindow('mask')


def circle_mask_on_random_image_in_path(path, target_size=None, r = None, extension=".jpg"):
    global accepted, winname, masks_done, rr

    numb = len([i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))])
    try:
        j = np.random.randint(numb)
    except:
        print(path)
        return


    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    if os.path.exists(path + '/mask/' + str(j) + '.jpg'):
        return
    if r == None and target_size!=None:
        rr = int(target_size[0]/10)
    else:
        rr = r
    ImName=str(j)+extension
    img = read_and_size(str(j), path=path, target_size=target_size, extension=extension)
    w, h, c = getWidthHeightChannels(img)
    if r == None and target_size == None:
        rr = int(w / 10)
        target_size=(w,h)
    show(img)

    accepted = False
    while (not accepted):
        accepted = True

        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2 = copy.deepcopy(img)
        cv2.drawContours(im2, contours, 0, (0, 255, 255), 2)

        show(im2)


    split_path = path.split("/")[:-1]


    repo_path = reduce((lambda x, y: x+'/'+y), split_path[:len(split_path)-3])
    if not os.path.isfile(repo_path + "/maskData.csv"):
        csvFile = open(repo_path+'/maskData.csv', 'w', newline="")
        writer = csv.writer(csvFile)
        writer.writerow(['patient', 'date', 'eye', 'name', 'width', 'height', 'x', 'y', 'r'])
        csvFile.close()

    csvFile = open(repo_path + '/maskData.csv', 'a', newline="")
    writer = csv.writer(csvFile)
    ls = split_path[-3:]
    ls.extend([ImName, target_size[0], target_size[1], xx, yy, rr])
    writer.writerow(ls)
    csvFile.close()
    cv2.imwrite(path + '/mask/' + ImName, mask)
    masks_done += 1
    print("masks: " + str(masks_done))
    cv2.destroyWindow('mask')

def mask_on_path(path):
    global accepted,winname,masks_done
    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    for i in range(len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])):
        if os.path.exists(path+'/mask/'+str(i)+'.jpg'):
            masks_done+=1
            continue
        img = read_and_size(str(i), path=path)

        show(img)
        #mask=cv2.bitwise_not(mask)

        accepted=False
        while(not accepted):
            accepted=True

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im2=copy.deepcopy(img)
            cv2.drawContours(im2, contours, 0, (0, 255, 0), 1)

            #ilo = cv2.bitwise_or(mask, img)
            #cv2.imshow('a', ilo)

            show(im2)

        cv2.imwrite(path+'/mask/'+str(i)+'.jpg', mask)
        masks_done+=1
        print("masks: "+str(masks_done))
        cv2.destroyWindow('a')
        cv2.destroyWindow('mask')




def square_circle_on_1_2_in_path(path):
    for i in range(6):
        im, im_cp = read_and_size_with_copy(str(i), path=path, scale=0.2)

        im=equalize_border_with_mean_or_val(im)

        h, w = np.shape(im)
        r = (int)(h / 8)
        img_sqd = square_circle_difference_filter_2d(im, r)
        #img_sqd = square_circle_minus_filter_2d(im, r)
        #img_sqd = circle_filter_2d(im)

        img_sqd = cv2.equalizeHist(img_sqd)

        ret, imt = cv2.threshold(img_sqd,250, 255, cv2.THRESH_BINARY)
        cv2.imshow('win2', im_cp)
        cv2.imshow('win32', img_sqd)
        show(imt)

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


def model_show_function(x):
    y=[]
    for i in range(len(x)-1):
        y.append(x[i+1])
    show(x[0], other_im=y)
