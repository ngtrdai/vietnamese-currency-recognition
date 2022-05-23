# @author: Nguyen Trong Dai
# @date: 23/05/2022
# @file: createDataset.py

import numpy as np
import cv2 as cv
from os import listdir
import pickle
from sklearn.preprocessing import LabelBinarizer

def captureData(labelName):
    camera = cv.VideoCapture(0)
    count = 96
    countFrame = 0
    label = labelName
    while count < 250:
        countFrame += 1
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv.resize(frame, dsize=None,fx=0.3,fy=0.3)
        cv.imshow('Camera', frame)
        if countFrame >= 10:
            countFrame = 0
            cv.imwrite('dataset/' + str(label) + '/' + str(label) + "_" + str(count) + ".png",frame)
            print(count)
            count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()

def saveData(rawFolder):
    kichThuocAnh = (128, 128)
    print("Bắt đầu xử lí...")
    images = []
    labels = []

    for folder in listdir(rawFolder):
        print("Folder=",folder)
        for file in listdir(rawFolder  + folder):
            print("File=", file)
            img = cv.imread(rawFolder  + folder +"/" + file)
            images.append(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB),dsize=(128,128)))
            labels.append(folder)
    images = np.array(images)
    labels = np.array(labels) #.reshape(-1,1)
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('./dataset/vietnamese_currency.data', 'wb')
    pickle.dump((images,labels), file)
    file.close()
    return 

def main():
    isCreate = False
    if isCreate:
        captureData("000000")
    else:
        saveData("./dataset/")

if __name__ == "__main__":
    main()