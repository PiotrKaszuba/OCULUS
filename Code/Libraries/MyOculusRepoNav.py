import os
from functools import reduce

import cv2
import numpy as np


def getPatientDateEye(image_path):
    image_path = image_path.split('/')[:-1]
    return image_path[0], image_path[1], image_path[2]


def getRepoPathAndImagePath(path):
    path = path.split('/')[:-1]
    repo = reduce((lambda x, y: x + '/' + y), path[:len(path) - 3]) + '/'
    image = reduce((lambda x, y: x + '/' + y), path[-3:]) + '/'
    return repo, image


def createImageInPath(path, name, image, override=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path + name):
        cv2.imwrite(path + name, image)
        return True
    if override:
        cv2.imwrite(path + name, image)
        return True
    return False


def execute_for_path(func, path, eye):
    if eye == 'left':
        func(path + 'left_eye_images/')
    if eye == 'right':
        func(path + 'right_eye_images/')
    if eye == 'both':
        func(path + 'right_eye_images/')
        func(path + 'left_eye_images/')


def all_path(func, start_path, eye=None):
    if eye != 'left' and eye != 'right':
        eye = 'both'

    patient = os.listdir(start_path)
    max = len([a for a in patient if not os.path.isfile(os.path.join(start_path, a))])

    obj = MyOculusRepoNav()

    obj.iterate_paths(func, start_path, eye, patient, max)


def random_path(start_path, eye=None):
    if eye != 'left' and eye != 'right':
        if np.random.randint(0, 2) == 0:
            eye = 'left'
        else:
            eye = 'right'

    patient = os.listdir(start_path)

    leng = len([f for f in patient if not os.path.isfile(os.path.join(start_path, f))])

    patient_path = patient[np.random.randint(0, leng)]

    date = os.listdir(start_path + patient_path + "/")

    date_path = date[np.random.randint(0, len(date))]

    if start_path == '':
        return patient_path + "/" + date_path + "/" + eye + "_eye_images/"
    else:
        return start_path + "/" + patient_path + "/" + date_path + "/" + eye + "_eye_images/"


class MyOculusRepoNav:

    def __init__(self):
        self.patients_done = 0

    def iterate_paths(self, func, start_path, eye, patient, max):

        for i in range(max):
            date = os.listdir(start_path + patient[i] + "/")

            for j in range(len(date)):
                path = start_path + patient[i] + '/' + date[j] + '/'
                execute_for_path(func, path, eye)

            self.patients_done += 1
            print("patients: " + str(self.patients_done))
