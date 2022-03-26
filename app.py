from operator import truediv
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
thres = 0.65 # Threshold to detect object

imagesPath = 'data/Images'
configPath = 'data/ssd_mobilenet_v3.pbtxt'
weightsPath = 'data/inference_graph.pb'
objectsPath = 'data/objects.names'
images = []
classImages = []
classNames = []
peopleInImg = []
uniqueID = 1
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

def drawBox(img,bbox,name):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),color=(255,255,0),thickness=3)
    cv2.rectangle(img,bbox,color=(255,255,0),thickness=3)
    cv2.putText(img,'tracking',(75,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.putText(img,str(name),(x+10,y+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)


def recognizePers(img):
    name = 'unkown'
    found = False
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classImages[matchIndex].upper()
            storeVisitors(name)
            found = searchImg(name)
            #y1,x2,y2,x1 = faceLoc
            #y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            #cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    return name,found

def track(img,name):
    #boxes_ids = tracker.update(detections)
    #for box_id in boxes_ids:
    #    x, y, w, h, id = box_id
    #    cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    #    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)        
    sucess,bbox = tracker.update(img)
    if sucess:
        for people in peopleInImg:
            if name == people[0]:
                people[1] = bbox
        drawBox(img,bbox,name)
    else:
        for people in peopleInImg:
            if name == people[0]:
                people.remove()
        cv2.putText(img,'lost',(75,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

def searchImg(name):
    if len(peopleInImg) > 0:
        for people in peopleInImg:
            if name == people[0]:
                return True
    return False

def setID():
    global uniqueID
    id_str = 'id' + str(uniqueID)
    uniqueID += 1
    return id_str


encodeListKnown = findEncodings(images)
print('Encoding Complete')

#starting video
cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4,720)
cap.set(10,70)
tracker = cv2.TrackerCSRT_create()

count =0
name = 'unkown'
tarckedImg = False

#looping to detect objects and faces
while True:
    timer = cv2.getTickCount()
    success,img = cap.read()

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)   
    
    if count%10 == 0:
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        cv2.putText(img,'detecting',(75,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    
    count+=1
    print(classIds,bbox)
    
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            #print(classIds.flatten()[0])
            #print(classId-1)
            #if count%60 == 0:
            if classId-1 == 0:
                name,found = recognizePers(img)
                if not found:
                    tracker.init(img,box)
                    if name == 'unkown':
                        name = setID()
                    peopleInImg.append([name,box])
                track(img,name)
                #cv2.rectangle(img,box,color=(0,0,255),thickness=3)
                #cv2.putText(img,str(name),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                #cv2.putText(img,str(round(confidence*100,2)),(box[0]+300,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

            if  classId-1 != 0:
                cv2.rectangle(img,box,color=(0,255,0),thickness=3)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+250,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            
    cv2.imshow('Object_detection',img)
    if cv2.waitKey(2) & 0xFF == 27:
            break
    
cap.release()
cv2.destroyWindow('Object_detection')