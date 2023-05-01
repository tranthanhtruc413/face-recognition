import RPi.GPIO as GPIO
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascade = 'Cascades/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascade)
font = cv2.FONT_HERSHEY_SIMPLEX

GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT)
id = 0
names = ['Tran_Thanh_Truc', '', '', '', '', '']

cap = cv2.VideoCapture(0)
cap.set(3, 640);
cap.set(4, 480);

minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int (minH))
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        print(str(id) + " => " + str(confidence))
        if (confidence < 80):
            id = names[id]
            confidence = " {0}".format(round(confidence))
            #GPIO.output(14, 1)
        else:
            id = "unknown"
            confidence = " {0}".format(round(confidence))
        
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
    cv2.imshow("Camera", img)
    #GPIO.output(14, 0)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\nPress ESC to exit...")
cap.release()
cv2.destroyAllWindows()
                    
    
