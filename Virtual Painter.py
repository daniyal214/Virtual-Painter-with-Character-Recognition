import cv2
import numpy as np
import os
import autopy
import HandTrackingModule as htm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
##################################################
camW, camH = 640,480
xp, yp = 0,0
imgCanvas = np.zeros((camH, camW, 3), np.uint8)
drawColor = (255, 0, 255)

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
##################################################

model = load_model('/home/daniyal214/Desktop/Data Science/Virtual Painter/new.h5')

image1 = cv2.imread('images.png')
image1 = cv2.resize(image1, (60,60))
image2 = cv2.imread('image2.png')
image2 = cv2.resize(image2, (640,100))
image3 = cv2.imread('71-mvqyYrnL._AC_SL1500_.png')
image3 = cv2.resize(image3, (60,60))
image4 = cv2.imread('pre.png')
image4 = cv2.flip(image4,1)
image4 = cv2.resize(image4, (200,60))

url = 'http://192.168.0.100:8080/video'
cap = cv2.VideoCapture(0)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(width, height)

cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectConf=0.85)

while True:
    success, img = cap.read()
    # cv2.circle(img, (camW//2, camH//2), 3, (0,255,0), 3)
    cv2.rectangle(img,(camW//2 - 100, camH//2 - 100), (camW//2 + 100, camH//2 + 100), (0,255,0),2)

    img[0:100, 0:640] = image2
    img[20:80, 20:80] = image1
    img[20:80, 560:620] = image3
    img[20:80, 220:420] = image4

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    inv_lst = []
    text_lst = []

    if len(lmList) != 0:
        xp, yp = 0, 0
        # print(lmList)
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]


        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[1] and fingers[2]:
            if y1<120:
                if 20 < x1 < 80:
                    drawColor = (0, 0, 0)

                elif 560 < x1 < 620:
                    drawColor = (255,0,0)

                elif 220 < x1 < 420:
                    imgInv2 = cv2.bitwise_not(imgInv)
                    # (camW // 2 - 100, camH // 2 - 100), (camW // 2 + 100, camH // 2 + 100)
                    imgInv2 = imgInv2[140:340,220:420]
                    cv2.imwrite('cropped.png', imgInv2)
                    inv_lst.append(imgInv2)



            cv2.rectangle(img, (x1,y1-25), (x2,y2-25), drawColor, cv2.FILLED)
            print('Selection Mode')

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, 25)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, 100)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 100)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, 25)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 25)

                xp, yp = x1, y1


            print('Drawing Mode')

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 0,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # fingers = detector.fingersUp()
    #
    # if fingers[1] and fingers[2] and fingers[3]:
    #
    # imgInv2 = cv2.bitwise_not(imgInv)
    if len(inv_lst) !=0:
        # cv2.imwrite('cropped.png', inv_lst[0])
        img2 = cv2.resize(inv_lst[0], (32, 32))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print(img.shape)

        cv2.imwrite('new2.png', img2)
        img_erosion = cv2.erode(img2, None, iterations=1)

        img_array = image.img_to_array(img_erosion)
        print(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        probs = model.predict(img_batch)
        prediction = probs.argmax(axis=1)
        pred_text = labelNames[prediction[0]]
        print(labelNames[prediction[0]])

        text = 'PREDICTION: {}'.format(pred_text)
        print(text)
        text_lst.append(text)

    if len(text_lst) !=0:
        text = text_lst[0]
        cv2.putText(img, text,(50,400), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 5)
        # img[20:80, 20:120] = text






    # cv2.imshow('Canvas',imgCanvas)
    # cv2.imshow('Inverse',imgInv)
    cv2.imshow('frame',img)

    cv2.waitKey(1)



























