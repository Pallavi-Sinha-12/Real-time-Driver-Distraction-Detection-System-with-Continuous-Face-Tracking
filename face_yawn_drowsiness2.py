#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import imutils
import time
import dlib
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
output_path = 'user/'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
yawn = "yawn.wav"
drowse = "drowse.wav"
facenot = "facenot.wav"
look2 = "look2.wav"


# In[2]:


EYE_AR_THRESH = 0.2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# In[3]:


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for x,y,w,h in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = output_path  + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('face cropper', face)
    else:
        print('face not found')
        pass
    if cv2.waitKey(1) == 13 or count == 100:
        break
cap.release()
cv2.destroyAllWindows()  
print("collecting samples")


# In[7]:


onlyfiles = [f for f in listdir(output_path) if isfile(join(output_path, f))]
Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles):
    image_path = output_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype = np.uint8))
    Labels.append(i)
Labels = np.asarray(Labels, dtype = np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("model trained")


# In[5]:


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def getting_landmarks(im):
    rects = detector(im,1)
    
    if len(rects)>1:
        return 'error'
    if len(rects)==0:
        return 'error'
    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])

def drow(image):
    shape = getting_landmarks(image)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear

def top_lip(landmarks):
    top_lip_pts=[]
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts,axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts=[]
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts,axis=0)
    return int(bottom_lip_mean[:,1])

def drow_yawn(image):
    shape = getting_landmarks(image)
    if shape == 'error':
        return 0,0
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    top_lip_center = top_lip(shape)
    bottom_lip_center = bottom_lip(shape)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return ear, lip_distance
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for x,y,w,h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi


# In[6]:


yawns = 0
yawn_status = False
pygame.init()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        if results[1] < 500:
            confidence = int(100*(1-(results[1])/300))
            display_string = str(confidence) + '% confident it is user'
        cv2.putText(image, display_string, (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120, 150), 2)
        if confidence>80:
            #FACE MATCHED SITUATION
            cv2.putText(image, "MATCHED", (350,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            #FACE NOT MATCHED
            cv2.putText(image, "NOT_MATCHED", (350,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            pygame.mixer.music.load(facenot)
            pygame.mixer.music.play()
        ear, lip_distance = drow_yawn(image)
        if ear<EYE_AR_THRESH:
            #DROWSINESS SITUATION
            cv2.putText(image, "DROWSINESS ALERT!", (10, 30),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pygame.mixer.music.load(drowse)
            pygame.mixer.music.play()
                
        prev_yawn_status = yawn_status
        if lip_distance > 25:
            yawn_status = True
            output_text = 'yawn count:' +str(yawns+1)
            if yawns>=2:
                #YAWN SITUATION- GREATER THAN OR EQUAL TO 3
                pygame.mixer.music.load(yawn)
                pygame.mixer.music.play()
            cv2.putText(frame,output_text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
        else:
            yawn_status = False
            
        if (prev_yawn_status == True and yawn_status == False):
            yawns+=1
        if(yawns==3):
            cv2.putText(image,'Yawned 3 times',(50,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face cropper", image)
    except:
        #FACE NOT FOUND SITUATION
        cv2.putText(image, "no face found",(220,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
        cv2.putText(image, "locked", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Face cropper", image)
        pygame.mixer.music.load(look2)
        pygame.mixer.music.play()
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




