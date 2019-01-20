import Code.Libraries.MyOculusCsvLib as mocl


def splitCsv(maskDataPath, csv1, csv2, howManyToNewCsv):
    list = mocl.getCsvList(maskDataPath, False)
    i = 0
    for row in list:
        if len(list) - i > howManyToNewCsv:
            mocl.writeToCsv(csv1, mocl.getMaskHeader(), row)
        else:
            mocl.writeToCsv(csv2, mocl.getMaskHeader(), row)
        i += 1


if __name__ == "__main__":
    repo_base = "../../Images/all/"
    csv1 = "../../Images/SharedMaskData/maskDataTrain800.csv"
    csv2 = "../../Images/SharedMaskData/maskDataTest55.csv"
    splitCsv(repo_base, csv1, csv2, 0)
