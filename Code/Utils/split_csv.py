import Code.Libraries.MyOculusCsvLib as mocl


def splitCsv(maskDataPath, name, csv1, csv2, howManyToNewCsv):
    list = mocl.getCsvList(maskDataPath, False, name=name)
    i = 0
    for row in list:
        if len(list) - i > howManyToNewCsv:
            if csv1 is not None:
                mocl.writeToCsv(csv1, mocl.getMaskHeader(), row)
        else:
            if csv2 is not None:
                mocl.writeToCsv(csv2, mocl.getMaskHeader(), row)
        i += 1


if __name__ == "__main__":
    repo_base = "../../SharedMaskData/"
    name = "maskData_3.csv"
    csv1 = "../../SharedMaskData/maskDataTrain700.csv"
    csv2 = "../../SharedMaskData/maskDataTest100.csv"
    splitCsv(repo_base, name, csv1, csv2, 100)
