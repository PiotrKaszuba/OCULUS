import os
import csv
import Code.Libraries.MyOculusImageLib as moil
import Code.Libraries.MyOculusRepoNav as morn


def getMaskHeader():
    return ['patient', 'date', 'eye', 'name', 'width', 'height', 'x', 'y', 'r']



def writeToCsv(path, header, row, overwrite=False):
    if (not os.path.isfile(path)) or overwrite:
        csvFile = open(path, 'w', newline="")
        writer = csv.writer(csvFile)
        writer.writerow(header)
        csvFile.close()

    if row is not None:
        csvFile = open(path, 'a', newline="")
        writer = csv.writer(csvFile)
        writer.writerow(row)
        csvFile.close()


def registerImageCsv(repo_path, image_path, image_name, image, function):
    patient, date, eye = morn.getPatientDateEye(image_path)
    width, height, channels = moil.getWidthHeightChannels(image)
    func_name = function.__name__
    header = ['patient', 'date', 'eye', 'name', 'width', 'height', 'channels', 'function']
    row = [patient, date, eye, image_name, width, height, channels, func_name]
    writeToCsv(repo_path + "imageData.csv", header, row)


def getCsvList(repo_path, image=True):
    list = []
    if image:
        name = "imageData.csv"
    else:
        name = "maskData.csv"
    with open(repo_path + name, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            list.append(row)
    return list


def checkIfExistsInCSV(patientDateEyeNameList, repo_path=None, list=None, image=True, returnTargetRow=False):
    print("Just checking CSV, dont worry")
    assert list is not None or repo_path is not None
    if list is None:
        list = getCsvList(repo_path, image=image)
    target = None
    for row in list:
        equal = True
        for j in range(4):
            if row[j] != patientDateEyeNameList[j]:
                equal = False
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
