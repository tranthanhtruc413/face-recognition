import cv2
import numpy as np
import os

from PIL import Image

img_dir = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

def getImageAndLabels(img_dir):
    imagePaths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
        
    return faceSamples, ids

print("\nFace training. please wait...")
faces, ids = getImageAndLabels(img_dir)
recognizer.train(faces, np.array(ids))

recognizer.write("trainer/trainer.yml")
print("\n{0} faces are learned.".format(len(np.unique(ids))))