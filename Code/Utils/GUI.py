import tkinter
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

import Code.Algorithms.Metrics as met
import Code.Libraries.MyOculusImageLib as moil
import Code.Preprocessing.MergeChannels as mc
import Code.Utils.CreateModel as cm


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
        self.label.pack()
        tkinter.Button(self.root, width=30, text="Załaduj zdjęcie", command=self.callback).pack()
        tkinter.Button(self.root, width=30, text="Oceń zdjęcie", command=self.make_prediction).pack()
        tkinter.Button(self.root, width=30, text="Oryginał", command=self.show_original).pack()
        self.root.resizable(False, False)

    def show_original(self):
        self.updateGuiImage(self.currentImg)

    def __init__(self):
        self.width = 600
        self.height = 450
        self.mer = mc.MergeChannels(equalize=True)
        self.loadModels()
        self.prepareGui()
        self.root.mainloop()

    def loadModels(self):
        self.Mod = cm.createOpticDiscModel("SAB700", gray=False, preprocessFunc=self.mer.Merge)
        self.Mod.model.predict(
            np.zeros(shape=(1, self.Mod.rowDim, self.Mod.colDim, self.Mod.channels), dtype=np.float32))
        self.ModAtrophy = cm.createAtophyModel("Gray50")
        self.ModAtrophy.model.predict(
            np.zeros(shape=(1, self.ModAtrophy.rowDim, self.ModAtrophy.colDim, self.ModAtrophy.channels),
                     dtype=np.float32))

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

    def AtrophyPrediction(self, roi):
        img = self.ModAtrophy.predict(roi)
        atrophyRate = met.atrophyRate(img)
        w, h, c = moil.getWidthHeightChannels(self.currentImg)
        img = cv2.resize(img, (round(160 * w / 600), round(160 * (w * 0.75) / 450)))
        img = moil.getBinaryThreshold(img)

        return atrophyRate, img

    def make_prediction(self):
        x, y, pred = self.OpticDiscPrediction()
        copy = self.currentImg.copy()
        drawCopy = self.currentImg.copy()
        drawCopy = moil.stackImageChannels(drawCopy)
        w, h, c = moil.getWidthHeightChannels(copy)
        xShift = int(80 * w / 600)
        yShift = int(80 * (w * 0.75) / 450)
        roi = moil.getRegionOfInterest(copy, x, y, xShift, yShift)

        atrophyRate, atrophyMap = self.AtrophyPrediction(roi)
        self.label.configure(text="Stopień zaniku (tylko faza tętniczo-żylna): " + str(atrophyRate))
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
        cv2.rectangle(drawCopy, (x-xShift, y-yShift) , (x+xShift, y+yShift), (127,0,127), int(5/1387*w))
        cv2.circle(drawCopy, (x, y), int(12 / 1387 * w), (127, 0, 127), thickness=int(5 / 1387 * w))
        met.draw(pred, drawCopy, thickness=int(4 / 1387 * w))
        self.updateGuiImage(drawCopy)

    def updateGuiImage(self, img):
        img = cv2.resize(img, (self.width, self.height))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.image.configure(image=imgtk)
        self.image.image = imgtk

    def callback(self):
        filename = filedialog.askopenfilename(title="Select file",
                                              filetypes=(("all files", "*.*"), ("jpeg files", "*.jpg")))

        if filename == '':
            return
        img = moil.read_and_size(name='', path=filename, extension='')
        self.currentImg = img
        self.label.configure(text="Stopień zaniku (tylko faza tętniczo-żylna): ")
        self.updateGuiImage(img)


GUI()
