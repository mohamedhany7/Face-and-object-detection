import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
thres = 0.6 # Threshold to detect object

imagesPath = 'data/Images'
configPath = 'data/ssd_mobilenet_v3.pbtxt'
weightsPath = 'data/inference_graph.pb'
objectsPath = 'data/objects.names'
images = []
classImages = []
classNames = []
imageList = os.listdir(imagesPath)

#geting objects names
with open(objectsPath,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#detecting the model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#importing images from file and creating names list
for i in imageList:
    img = cv2.imread(f'{imagesPath}/{i}')
    images.append(img)
    classImages.append(os.path.splitext(i)[0])
print(classImages)

#function to get images's encodings 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def storeVisitors(name):
    with open('data/visitors.csv','r+') as f:
        visitorList = f.readlines()
        vlist = []
        for line in visitorList:
            entry = line.split(',')
            vlist.append(entry[0])
        if name not in vlist:
            time = datetime.now()
            str = time.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{str}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

#starting video
cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4,720)
cap.set(10,70)

#looping to detect objects and faces
while True:
    success,img = cap.read()
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            #print(classIds.flatten()[0])
            #if classIds.flatten()[0] == 1:
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
            #if classId == 1:
            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                matchIndex = np.argmin(faceDis)
        
                if matches[matchIndex]:
                    name = classImages[matchIndex].upper()
                    storeVisitors(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #else:
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            print(classNames[classId-1].upper())
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+250,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Object_detection',img)
    if cv2.waitKey(2) & 0xFF == 27:
            break
    
cap.release()
cv2.destroyWindow('Object_detection')