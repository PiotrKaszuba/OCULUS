import cv2
import numpy as np
import math
import copy
import os
import shutil
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
        r=(int)(h/10)
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
        r=(int)(h/10)
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
    for i in range (len(patient)):
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
        if np.random.randint(0,2) ==1:
            eye='right'


    patient = None
    global image_path
    if start_path==None:
        start_path=image_path

    patient = os.listdir(start_path)


    patient_path = patient[np.random.randint(0,len(patient))]

    date = os.listdir(start_path+patient_path+"/")

    date_path = date[np.random.randint(0,len(date))]

    if start_path=='':
        return patient_path+"/"+date_path+"/"+eye+"_eye_images/"
    else:
        return start_path+"/"+patient_path + "/" + date_path + "/" + eye + "_eye_images/"

def get_number_of_images_in_path():
    global image_path
    return len(os.listdir(image_path))


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

def read_and_size( name, path=None, extension='.jpg', scale=0.2, mode=0, modify_shape=True, h_div_w = 0, modify_height=False, not_modified_wanted_value=0):
    global image_path
    if path == None:
        path = image_path
    im = cv2.imread(path+name+extension, mode)
    if scale >0 :
        im = cv2.resize(im, (0,0), fx=scale, fy=scale)
    if h_div_w>0:
        im = modify_h_div_w(im,h_div_w,modify_height,not_modified_wanted_value)
    if modify_shape:
        load_h_w_from_img(im)
    return im

def read_and_size_with_copy( name, path=None, extension='.jpg', scale=0.2, mode=0, modify_shape=True, h_div_w = 0, modify_height=False, not_modified_wanted_value=0):
    global image_path
    if path == None:
        path = image_path
    im = cv2.imread(path+name+extension, mode)
    if scale > 0:
        im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    if h_div_w > 0:
        im = modify_h_div_w(im, h_div_w, modify_height, not_modified_wanted_value)
    im_copy = copy.deepcopy(im)
    if modify_shape:
        load_h_w_from_img(im)
    return im, im_copy

def trackback_callback(x):
    global track_val
    track_val = x

def show(im, function_on_im=None, *args, other_im = [], function_on_other=None, print=True, window=None):
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
    h, w = np.shape(img)
    global height
    global width
    height = h
    width = w

def init(win='win', im_path = 'Images/all/', start_path='Images/all/', mouse_f=measure_size, w=518, h=392):
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


def mask_on_path(path):
    global accepted,winname,masks_done
    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    for i in range( len(os.listdir(path))-1):
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
    for i in range(2):
        im, im_cp = read_and_size_with_copy(str(i), path=path, scale=0.15)

        im=equalize_border_with_mean_or_val(im)

        #img_sqd = square_circle_minus_filter_2d(im)
        img_sqd = circle_filter_2d(im)

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