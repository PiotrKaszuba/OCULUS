import os
import shutil


# lists all patients in local directory
def string_patients(path):
    list = os.listdir(path)
    string = ''
    for item in sorted(list):
        if os.path.isfile(os.path.join(path, item)):
            continue
        string += item + '\n'
    return string


# returns same patient lines in filename and local directory
def same_lines(filename, path_to_txt, path_to_local):
    string_local = string_patients(path_to_local)
    file = open(path_to_txt + filename + '.txt', 'r')
    string_file = file.read()

    local_list = string_local.split('\n')
    file_list = sorted(string_file.split('\n'))
    string = ''
    i = 0
    imax = len(local_list)
    j = 0
    jmax = len(file_list)

    while i != imax and j != jmax:
        if local_list[i] > file_list[j]:
            j += 1
        if local_list[i] < file_list[j]:
            i += 1
        if local_list[i] == file_list[j]:
            string += local_list[i] + '\n'
            i += 1
            j += 1

    return string


def delete_directories(lines_of_directories, path):
    list = lines_of_directories.split('\n')
    for item in list:
        shutil.rmtree(path + item)


def to_txt(string, name, path_to_txt):
    file = open(path_to_txt + name + '.txt', 'w')
    file.write(string)


def save_patients_to_txt():
    path = '../../Images/data/'
    string = string_patients(path)

    filename = 'Piotr'
    path_to_patients = '../../Images/patients/'

    to_txt(string, filename, path_to_patients)


def clear_same_dirs():
    path = '../../Images/data2/'
    path_to_patients = '../../Images/patients/'
    filename = 'Piotr'

    delete_directories(same_lines(filename, path_to_txt=path_to_patients, path_to_local=path), path)
