import cv2
import numpy as np
import math
import copy
import os
import shutil
import re
import csv
from functools import reduce
import random
import ctypes
import textwrap
from difflib import SequenceMatcher
import nltk
import tkinter
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




def execute_for_path(func, path, eye, data_once_per_patient_eye_tuple):
    if eye == 'left':
        data = func(path + 'left_eye_images/')
    if eye == 'right':
        data = func(path + 'right_eye_images/')
    if eye == 'both':
        data2 = func(path + 'right_eye_images/')
        data = func(path + 'left_eye_images/')

        if data_once_per_patient_eye_tuple[1]:
            data = save_data(data, data2, True)
    return data

def save_data(dictionary, data, collect_data):
    if collect_data:
        for key, value in data.items():
            old = dictionary.get(key)
            if old is None:
                old = []
            for item in value:
                old.append(item)
            dictionary[key] = old
    return dictionary

def iterate_paths(func, start_path, eye, data_once_per_patient_eye_tuple, collect_data, patient, max, dictionary):
    global patients_done
    for i in range (max):
        date = os.listdir(start_path + patient[i] + "/")

        for j in range(len(date)):
            path = start_path+patient[i]+'/'+date[j]+'/'
            data = execute_for_path(func, path, eye, data_once_per_patient_eye_tuple)
            dictionary = save_data(dictionary, data, collect_data)
            if data_once_per_patient_eye_tuple[0]:
                break
        patients_done+=1
        print("patients: " +str(patients_done))
    return dictionary

def count_n_grams(value):
    for val in value:
        splitted = re.findall(r"[\w']+", val)
        ngrams = nltk.ngrams(splitted, 4)

def split_to_words(temp_dict, val):
    splitted = re.findall(r"[\w']+", val)
    for split in splitted:
        temp_dict = increment_dict(temp_dict, split)
    return temp_dict

def increment_dict(temp_dict, val):
    old_count = temp_dict.get(val)
    if old_count == None:
        old_count = 0
    temp_dict[val] = old_count + 1
    return temp_dict


def map_chain(mapping_dict, word, word2, rate):
    word_mapping = mapping_dict.get(word)
    while word_mapping is not None:
        word = word_mapping[0]
        word_mapping = mapping_dict.get(word)
    mapping_dict[word2] = (word, rate)
    return mapping_dict

def merge_mapping(sorted_list, ratio):
    mapping_dict = {}
    for i in range(len(sorted_list)):
        word = sorted_list[i][0]
        print("merge mapping: " + str(i))
        for j in range(i+1, len(sorted_list)):
            word2 = sorted_list[j][0]
            rate = similar(word, word2)
            need_ratio = ratio + 0.015 * len(word2)
            if rate > need_ratio:
                temp_mapping = mapping_dict.get(word2)
                if (temp_mapping is not None and rate > temp_mapping[1]) or temp_mapping is None:
                    mapping_dict = map_chain(mapping_dict, word, word2, rate)
    return mapping_dict

def save_mapping_to_csv(merge_mapping):
    sorted_by_value = sorted(merge_mapping.items(), key=lambda kv: kv[1][1], reverse=True)
    csvFile = open("merge_mapping.csv", 'w', newline="")
    writer = csv.writer(csvFile)
    writer.writerow(["mapped", "to", "ratio"])
    for value in sorted_by_value:
        writer.writerow([value[0], value[1][0], value[1][1]])
    csvFile.close()

def process_data(dictionary, collect_data, data_process_keys):
    if collect_data:
        if data_process_keys != None:
            key_names = ["ignore_keys", "merge_keys", "n_gram_keys", "split_to_words_keys", "merge_mapping_keys"]
            dict_keys = {}
            for name in key_names:
                temp = data_process_keys.get(name)
                if temp is None:
                    temp = []
                dict_keys[name] = temp

            total_dict = {}
            #type data and data
            for key, value in dictionary.items():
                operations_done = []
                if key in dict_keys.get("ignore_keys"):
                    continue
                #if key in dict_keys.get("n_gram_keys"):
                #    count_n_grams(value)
                temp_dict = {}
                #val is piece of data
                for val in value:
                    if key in dict_keys.get("split_to_words_keys"):
                        #increments inside
                        temp_dict = split_to_words(temp_dict, val)
                        operations_done.append("split_to_words_keys")
                    else:
                        temp_dict = increment_dict(temp_dict, val)


                sorted_by_value = sorted(temp_dict.items(), key=lambda kv: kv[1], reverse=True)
                if key in dict_keys.get("merge_mapping_keys"):
                    merge_mapping_dict = merge_mapping(sorted_by_value, data_process_keys.get("merge_mapping_keys_ratio"))
                    operations_done.append("merge_mapping_keys")
                    if data_process_keys.get("merge_mapping_keys_save"):
                        save_mapping_to_csv(merge_mapping_dict)
                total_dict[key] = sorted_by_value
            print_total_dict(total_dict)
            return total_dict
    return dictionary

def all_path(func,start_path=None, eye=None, data_once_per_patient_eye_tuple=(False, False), collect_data=False, data_process_keys= None):

    if eye != 'left' and eye != 'right':
        eye='both'

    if start_path==None:
        start_path=image_path

    dictionary = {}
    patient = os.listdir(start_path)
    max = len([a for a in patient if not os.path.isfile(os.path.join(start_path,a))])

    dictionary = iterate_paths(func, start_path, eye, data_once_per_patient_eye_tuple, collect_data, patient, max, dictionary)
    dictionary = process_data(dictionary,collect_data,data_process_keys)

def print_total_dict(dict, key=None):
    print("Counts:")
    if key==None:
        for key, value in dict.items():
            print(key)
            for val, count in value:
                print(val+": " + str(count))
    else:
        print(key)
        for val, count in dict.get(key):
            print(val + ": " + str(count))

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
def getScreenSize():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
def concat_images(imList, imH=None,imW=None):
    w, h = getScreenSize()
    w = w+50
    a,b,c = getWidthHeightChannels(imList[0])
    if imH==None:
        imH = b
    if imW==None:
        imW = a
    cols = math.floor(w/imW)
    images = len(imList)
    rows = math.ceil(images/cols)
    new_img = np.zeros(shape=(imH*rows, imW*cols), dtype=np.uint8)
    for i in range(images):

        currRow = math.floor(i/cols)
        currCol = i - currRow * cols
        new_img[currRow*imH:currRow*imH+imH, currCol*imW:currCol*imW+imW] = imList[i]

    return new_img

def show_joined_print(images, text, window=None):
    global winname
    if window == None:
        window = winname
    key = None
    toprint=''
    for key, val in text.items():
        toprint+= val[0]
    print('\n'.join(textwrap.wrap(toprint, 180, break_long_words=False)))
    img = concat_images(images)
    while 1:
        cv2.imshow(window, img)

        key =cv2.waitKey(30)
        if key == ord('q'):
            break


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

def get_description(path):
    names = ["description1.txt",  "correct_icd_code.txt","age.txt", "sex.txt"]

    dict = {}
    for name in names:


        file = open(os.path.join(path,name), 'r', encoding='utf8')

        text = [file.read().lower()]
        if name == names[1]:
            text[0] = text[0][2:]
        dict[name] = text
        #text += file.read()+'\n'

    return dict

def get_description_full_path(path, imSize=(240,180), showImages=False):
    repo, image = getRepoPathAndImagePath(path)
    patient, date, eye=getPatientDateEye(image)
    list =[]
    listWait=[]
    if showImages:
        for a in os.listdir(path):
            if not os.path.isfile(os.path.join(path, a)):
                continue


            im = read_and_size(a, path, target_size=imSize, extension='')
            cv2.putText(im, str(a), (5, math.floor(np.shape(im)[0])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))
            if (int(a[:-4]) > 9):
                listWait.append(im)
            else:
                list.append(im)
        for ele in listWait:
            list.append(ele)
    dict = get_description(os.path.join(repo,patient,date))
    #show_joined_print(list, dict)
    return dict


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
        if os.path.isfile(os.path.join(path, item)):
            continue
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

def createFromAllPathImageAfterFunction(old_repo_path, new_repo_path, function, target_size=None, eye=None, override=False, extension='.jpg', onlyMasked=False):
    success = 0
    fail = 0
    if eye != 'left' and eye != 'right':
        eye='both'

    patient = None

    iter = 0
    patient = os.listdir(old_repo_path)
    for i in range (len([a for a in patient if not os.path.isfile(os.path.join(old_repo_path,a))])):

        date = os.listdir(old_repo_path + patient[i] + "/")

        for j in range(len(date)):
            if eye == 'left':
                t1, t2 = createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/', new_repo_path,function,target_size,override, extension, onlyMasked)
                success+=t1
                fail+=t2
            if eye == 'right':
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'right_eye_images/', new_repo_path,function,target_size,override, extension, onlyMasked)
                success += t1
                fail += t2
            if eye == 'both':
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'left_eye_images/', new_repo_path,function,target_size,override, extension, onlyMasked)
                success += t1
                fail += t2
                t1, t2 =createImagesInRepoAfterFunctionOnPath(old_repo_path+patient[i]+'/'+date[j]+'/'+'right_eye_images/', new_repo_path,function,target_size,override, extension, onlyMasked)
                success += t1
                fail += t2
        iter += 1
        print("Patients finished: " + str(iter))

    print("Images created: " + str(success) + ", attempts failed to create: " + str(fail))

def copyMaskData(old_repo_path, new_repo_path):
    shutil.copyfile(old_repo_path+'maskData.csv', new_repo_path+'maskData.csv')

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



def createImagesInRepoAfterFunctionOnPath(path, new_repo_path, function, target_size, override=False, extension='.jpg', onlyMasked = False):
    success = 0
    fail = 0
    repo_path, image_path = getRepoPathAndImagePath(path)

    base2, name2 = getBaseRepoPathAndRepoName(new_repo_path)
    createImagesRepo(base2+'/', name2)
    new_path = new_repo_path+image_path

    maskList=None
    if onlyMasked:
        maskList = getCsvList(repo_path, False)

    for a in os.listdir(path):
        if not os.path.isfile(os.path.join(path, a)):
            continue
        patient, date, eye = getPatientDateEye(image_path)

        if onlyMasked and not os.path.isfile(os.path.join(path+'mask/', a)) and not checkIfExistsInCSV([patient, date,eye,a],list=maskList,image=False):
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
    createImagesRepo(base+'/',name)
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

def checkIfExistsInCSV(patientDateEyeNameList, repo_path=None, list= None, image=True, returnTargetRow=False):
    print("Just checking CSV, dont worry")
    assert list is not None or repo_path is not None
    if list is None:
        list = getCsvList(repo_path, image=image)
    target = None
    for row in list:
        equal = True
        for j in range(4):
            if row[j] != patientDateEyeNameList[j]:
                equal=False
                break
        if equal:
            target = row
            break
    if returnTargetRow:
        return target
    else:
        if target is None:
            return False
        else:
            return True


def createMaskFromCsv(repo_path, imageRow, list = None, override=False):
    if list is None:
        list = getCsvList(repo_path, image=False)

    target = checkIfExistsInCSV(imageRow, repo_path=repo_path, list=list, image=False, returnTargetRow=True)
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
    iter=0
    with open(repo_path + "imageData.csv", 'r') as file:
        reader = csv.reader(file)
        next(reader,None)
        for row in reader:
            if createMaskFromCsv(repo_path,row,list,True):
                success+=1
            else:
                fail+=1
            iter+=1
            print("Images looped: " + str(iter))
        file.close()
    print("Masks created: "+str(success)+ ", failed to create: " +str(fail))


def circle_mask_on_path(path, target_size=None, r = None,extension=".jpg", check_csv=True, list=None):
    global accepted, winname, masks_done, rr
    if r == None and target_size !=None:
        rr = int(target_size[0]/10)
    else:
        rr = r
    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    skipped=0
    for ii in range(len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])):
        i = ii +skipped
        ImName = str(i) + extension
        while not os.path.exists(os.path.join(path, ImName)):
            skipped += 1
            i = ii + skipped
            ImName = str(i) + extension

        tempName = path + '/mask/' + ImName
        if os.path.exists(tempName):
            print("Path exists (" + tempName + ")")
            masks_done += 1
            continue

        if check_csv:
            paths = getRepoPathAndImagePath(path)
            row = paths[1].split("/")[:-1]
            row.append(ImName)
            if checkIfExistsInCSV(row, paths[0], list, False):
                print("In CSV exists (" + tempName + ")")
                continue


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


def circle_mask_on_random_image_in_path(path, target_size=None, r = None, extension=".jpg", check_csv=True, list=None):
    global accepted, winname, masks_done, rr, random

    numb = len([i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))])
    temp = ([a for a in os.listdir(path) if os.path.isfile(os.path.join(path, a))])
    try:
        j = np.random.randint(numb)
    except:
        print(path+", numb: "+str(numb))
        return

    ImName = random.choice(temp)
    if not os.path.exists(path+'/mask'):
        os.makedirs(path+'/mask')
    tempName = path + '/mask/' + ImName
    if os.path.exists(tempName):
        print("Path exists (" + tempName + ")")
        return
    if check_csv:
        paths = getRepoPathAndImagePath(path)
        row = paths[1].split("/")[:-1]
        row.append(ImName)
        if checkIfExistsInCSV(row, paths[0], list, False):
            print("In CSV exists (" + tempName + ")")
            return

    if r == None and target_size!=None:
        rr = int(target_size[0]/10)
    else:
        rr = r

    img = read_and_size(ImName, path=path, target_size=target_size, extension='')
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
'''def mergeListSimilarKeysTuples(dict, ratio=0.88):
    new_dict = []
    for i in range(len(dict)):
        key = dict[i][0]
        sum = dict[i][1]
        print(str(i))
        for j in range(i+1, len(dict)):
            key2=dict[j][0]
            if key == key2:
                continue
            rate = similar(key, key2)
            if rate > ratio:
                sum += dict[j][1]
                dict[j] = (key2, 0)
        new_dict.append((key, sum))
    return new_dict
'''
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
def model_show_function(x):
    y=[]
    for i in range(len(x)-1):
        y.append(x[i+1])
    show(x[0], other_im=y)

def user_gui_accept(row):


    return True

def choose_mappings(path_read="merge_mapping.csv", path_write = "accepted_mapping.csv", header=["mapped", "to", "ratio"]):
    root = tkinter.Tk()
    root.geometry("400x200")
    root.title = "accept?"
    global accept_row, var_mapped, var_to, var_ratio


    csv_read = open(path_read, 'r', newline="")
    accept_reader = csv.reader(csv_read)
    next(accept_reader, None)
    accept_row = next(accept_reader, None)

    var_mapped = tkinter.StringVar()
    var_to = tkinter.StringVar()
    var_ratio = tkinter.StringVar()
    mapped_label = tkinter.Label(root, textvariable=var_mapped, relief=tkinter.RAISED )
    mapped_label.pack()
    to_label = tkinter.Label(root, textvariable=var_to, relief=tkinter.RAISED)
    to_label.pack()
    ratio_label = tkinter.Label(root, textvariable=var_ratio, relief=tkinter.RAISED)
    ratio_label.pack()
    def reload_gui():
        global accept_row, var_mapped, var_to, var_ratio
        var_mapped.set("Mapowany: " +accept_row[0])
        var_to.set("Do bazowego: " +accept_row[1])
        var_ratio.set("Ratio: "  + accept_row[2])
    def new_row():
        global accept_row
        accept_row = next(accept_reader, None)


    def accept():
        writeToCsv(path_write, header, accept_row)
        new_row()
        reload_gui()

    def reject():
        new_row()
        reload_gui()

    reload_gui()
    accept_button = tkinter.Button(root, text="accept", command=accept, height=4, width=20)
    reject_button = tkinter.Button(root, text="reject", command=reject, height=4, width=20)

    accept_button.pack()
    reject_button.pack()


    root.mainloop()
    csv_read.close()

