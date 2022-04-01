from operator import truediv
import cv2
import os
from classes import Person
from functions import *

thres = 0.60 # Threshold to detect object
imagesPath = 'data/Images' 
configPath = 'data/ssd_mobilenet_v3.pbtxt'
weightsPath = 'data/inference_graph.pb'
objectsPath = 'data/objects.names'
classNames = []
trackings = []
persons = []
uniqueID = 1
frames = -1
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
    pers = Person(img,os.path.splitext(i)[0])
    persons.append(pers)
print('successfully created ' + str(len(persons)) + ' persons')

encodeListKnown = findEncodings(persons)
print('Encoding Complete')

#starting video
cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4,720)
cap.set(10,70)

print(cv2.getVersionString())
#tracker = cv2.TrackerCSRT_create()
trackerType = "CSRT"
#cv2.MultiTracker()
#multiTracker = cv2.MultiTracker_create()
multiTracker = cv2.legacy.MultiTracker_create()
#multiTracker=cv.MultiTracker()

#Main loop for detection
while True:
    frames+=1
    if frames%10 == 0:           #detecting every 10 frames
        timer = cv2.getTickCount()
        success,img = cap.read()

        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)   
        
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        cv2.putText(img,'detecting',(600,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        print(classIds,bbox)
        
        if len(classIds) != 0:          #checking there is an object in the frame
            update_tracking(img,multiTracker,trackings)
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                if classId-1 == 0:      #checking it's a person or not
                    recognized = recognizePers(img,box,encodeListKnown,trackings,persons,multiTracker)
                    if recognized == False:
                        drawBox(img,box,"person")
                    
                #elif classId-1 != 0:
                    #cv2.rectangle(img,box,color=(0,255,0),thickness=3)
                    #cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    #cv2.putText(img,str(round(confidence*100,2)),(box[0]+250,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                
        cv2.imshow('AI Camera',img)
        if cv2.waitKey(2) & 0xFF == 27:
                break
    
cap.release()
cv2.destroyWindow('AI Camera')