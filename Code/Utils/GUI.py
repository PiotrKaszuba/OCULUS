import tkinter
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

import Code.Algorithms.Metrics as met
import Code.Libraries.MyOculusCsvLib as mocl
import Code.Libraries.MyOculusImageLib as moil
import Code.Preprocessing.MergeChannels as mc
import Code.Utils.CreateModel as cm


# import easygui

class GUI:
    def prepareGui(self):
        self.root = tkinter.Tk()
        self.root.title("Oculus")

        img = np.zeros(shape=(self.height, self.width, 3), dtype=np.uint8)
        self.currentImg = img
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        self.image = tkinter.Label(self.root, image=imgtk)
        self.image.pack()
        self.image.config(height=self.height, width=self.width)
        self.label = tkinter.Label(self.root, text='Stopień zaniku (tylko faza tętniczo-żylna) : ')
        self.labelExit = tkinter.Label(self.root, text='Przesunięcie naczyń (faza tętniczo-żylna lub późna) : ')
        self.label.pack()
        self.labelExit.pack()
        tkinter.Button(self.root, width=30, text="Załaduj zdjęcie", command=self.callback).pack()
        tkinter.Button(self.root, width=30, text="Oceń zdjęcie", command=self.make_prediction).pack()
        tkinter.Button(self.root, width=30, text="Oryginał", command=self.show_original).pack()
        tkinter.Button(self.root, width=30, text="Zapisz do CSV", command=self.save_to_csv).pack()
        self.root.resizable(False, False)

    def show_original(self):
        self.updateGuiImage(self.currentImg)

    def __init__(self):
        self.width = 600
        self.height = 450
        self.path = None
        self.predicted = False
        self.x = -1
        self.y = -1
        self.atrophyRate = -1
        self.distance = -1
        self.xOut = -1
        self.yOut = -1
        self.loadModels()
        self.prepareGui()
        self.root.mainloop()

    def save_to_csv(self):
        if self.path is None:
            return
        if self.predicted == False:
            return
        mocl.writeToCsv("output.csv", mocl.getOutputHeader(),
                        [self.path, self.x, self.y, self.xOut, self.yOut, self.atrophyRate])

    def loadModels(self):
        self.mer = mc.MergeChannels(equalize=True)
        self.Mod = cm.createOpticDiscModel("SAB700_NODECAY", gray=False, preprocessFunc=self.mer.Merge)
        self.Mod.model.predict(
            np.zeros(shape=(1, self.Mod.rowDim, self.Mod.colDim, self.Mod.channels), dtype=np.float32))
        self.ModAtrophy = cm.createAtophyModel("Gray50")
        self.ModAtrophy.model.predict(
            np.zeros(shape=(1, self.ModAtrophy.rowDim, self.ModAtrophy.colDim, self.ModAtrophy.channels),
                     dtype=np.float32))
        self.ModExit = cm.createExitModel("Gray125")
        self.ModExit.model.predict(
            np.zeros(shape=(1, self.ModExit.rowDim, self.ModExit.colDim, self.ModExit.channels), dtype=np.float32))

    def OpticDiscPrediction(self):
        img = self.Mod.predict(self.currentImg)

        # img = moil.stackImageChannels(img)

        # resizing prediction
        w, h, c = moil.getWidthHeightChannels(self.currentImg)
        img = cv2.resize(img, (w, h))

        # getting coords
        x, y = met.getCenter(img, w, h)
        x = int(x)
        y = int(y)
        return x, y, img

    def ExitPrediction(self, roi, xExitShift, yExitShift, xRef, yRef):

        img = self.ModExit.predict(roi)

        img = cv2.resize(img, (xExitShift * 2, yExitShift * 2))

        x, y = met.getCenter(img, xExitShift * 2, yExitShift * 2)
        x = int(x)
        y = int(y)
        return x + xRef - xExitShift, y + yRef - yExitShift

    def AtrophyPrediction(self, roi):
        img = self.ModAtrophy.predict(roi)
        atrophyRate = met.atrophyRate(img)
        w, h, c = moil.getWidthHeightChannels(self.currentImg)
        img = cv2.resize(img, (round(160 * w / 600), round(160 * (w * 0.75) / 450)))
        img = moil.getBinaryThreshold(img)

        return atrophyRate, img

    def make_prediction(self):
        x, y, pred = self.OpticDiscPrediction()
        self.x = x
        self.y = y
        copy = self.currentImg.copy()
        drawCopy = self.currentImg.copy()
        drawCopy = moil.stackImageChannels(drawCopy)
        w, h, c = moil.getWidthHeightChannels(copy)
        xShift = int(80 * w / 600)
        yShift = int(80 * (w * 0.75) / 450)

        xExitShift = int(40 * w / 600)
        yExitShift = int(40 * (w * 0.75) / 450)
        roi = moil.getRegionOfInterest(copy, x, y, xShift, yShift)
        roiExit = moil.getRegionOfInterest(copy, x, y, xExitShift, yExitShift)
        atrophyRate, atrophyMap = self.AtrophyPrediction(roi)
        self.atrophyRate = atrophyRate
        self.label.configure(text="Stopień zaniku (tylko faza tętniczo-żylna): " + str(atrophyRate))

        xExit, yExit = self.ExitPrediction(roiExit, xExitShift, yExitShift, x, y)
        self.xOut = xExit
        self.yOut = yExit
        dist = np.linalg.norm(
            np.asarray([xExit / w * 600, yExit / (w * 0.75) * 450]) - np.asarray([x / w * 600, y / (w * 0.75) * 450]))
        if dist > 16:
            self.labelExit.configure(
                text='Przesunięcie naczyń (faza tętniczo-żylna lub późna) : ' + str(dist) + ', ZNACZNE!')
        else:
            self.labelExit.configure(
                text='Przesunięcie naczyń (faza tętniczo-żylna lub późna) : ' + str(dist))
        wA, hA, cA = moil.getWidthHeightChannels(atrophyMap)

        mask = np.zeros((h, w), drawCopy.dtype)
        mask = moil.addToRegionOfInterest(mask, x, y, round(wA / 2 + 0.00001), round(hA / 2 + 0.00001), atrophyMap)

        # mask[y-round(hA/2+0.00001):y+round(hA/2+0.00001), x-round(wA/2+0.00001):x+round(wA/2+0.00001)] = atrophyMap
        redImg = np.zeros(drawCopy.shape, drawCopy.dtype)
        redImg[:, :] = (255, 0, 0)
        redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
        drawCopy = cv2.addWeighted(redMask, 1, drawCopy, 1, 0)

        # moil.show(atrophyMap)
        # drawCopy[mask] = (255, 0, 0)
        cv2.rectangle(drawCopy, (x - xShift, y - yShift), (x + xShift, y + yShift), (127, 0, 127), int(5 / 1387 * w))
        cv2.circle(drawCopy, (x, y), int(12 / 1387 * w), (127, 0, 127), thickness=int(5 / 1387 * w))

        met.draw(pred, drawCopy, thickness=int(4 / 1387 * w))
        cv2.circle(drawCopy, (xExit, yExit), int(12 / 1387 * w), (0, 127, 0), thickness=int(5 / 1387 * w))
        self.updateGuiImage(drawCopy)
        self.predicted = True

    def updateGuiImage(self, img):
        img = cv2.resize(img, (self.width, self.height))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.image.configure(image=imgtk)
        self.image.image = imgtk

    def callback(self):

        filename = filedialog.askopenfilename(title="Select file",
                                              filetypes=(("all files", "*.*"), ("jpeg files", "*.jpg")))

        # filename = easygui.fileopenbox()
        self.root.update()
        if filename == '':
            return
        img = moil.read_and_size(name='', path=filename, extension='')
        self.currentImg = img
        self.x = -1
        self.y = -1
        self.atrophyRate = -1
        self.distance = -1
        self.xOut = -1
        self.yOut = -1
        self.predicted = False
        self.path = filename
        self.label.configure(text="Stopień zaniku (tylko faza tętniczo-żylna): ")
        self.updateGuiImage(img)


GUI()
