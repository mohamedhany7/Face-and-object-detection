from operator import truediv
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from classes import Person

#function to get images's encodings 
def findEncodings(persons):
    encodeList = []
    for img in persons:
        img = cv2.cvtColor(img.get_image(), cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def recognizePers(img,box,encodeListKnown,trackings,persons):
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    if len(facesCurFrame) >0:
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis)
            print(faceDis)

            if matches[matchIndex]:
                print(faceDis[matchIndex])
                if  faceDis[matchIndex]<0.6: #check if the person is known
                    name = persons[matchIndex].get_name().upper()
                    checkTracking(img,name,box,trackings,persons,faceLoc)
                    #drawBox(img,faceLoc,name,'red')
                    return True

            else:           #person is unkown
                name = 'unknow person'
                #drawBox(img,faceLoc,name.upper(),'red')
                takeScreenshot(img,box)
                return False
    return False

def checkTracking(img,name,box,trackings,persons,faceLoc):
    isUpdating = False
    if len(trackings)>0:
        for pers in trackings:
            if pers.get_name() == name:
                if pers.get_isTracked:
                    #update_tracking(img,name,box,pers,trackings,faceLoc,multiTracker)
                    isUpdating = True
    if isUpdating == False:
        track(img,name,box,trackings)


def track(img,name,box,trackings):
    tracker = cv2.TrackerKCF_create()
    pers = Person(img,name,box,True,tracker)
    trackings.append(pers)
    tracker.init(img,box)
    #multiTracker.add(cv2.legacy.TrackerCSRT_create(), img, box)

def update_tracking(img,multiTracker,trackings):
    success, boxes = multiTracker.update(img)
    if success:
        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(img, p1, p2, (255,0,255), 2, 1)
            #print(trackings[i].get_name())
            cv2.putText(img,trackings[i].get_name(),(int(newbox[0])+10,int(newbox[3])+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    else:
        #pers.lost()
        #trackings.remove(pers)
        cv2.putText(img,'lost',(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)


def update_tracking2(img,trackings):
    for pers in trackings:
        success = pers.updateTracker(img)
        if success:
            newbox = pers.get_bbox()
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(img, p1, p2, (255,0,255), 2, 1)
            #print(trackings[i].get_name())
            cv2.putText(img,pers.get_name(),(int(newbox[0])+10,int(newbox[3])+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            print('tracking ', pers.get_name())
        else:
            pers.lost()
            trackings.remove(pers)
            print('lost ', pers.get_name())
            cv2.putText(img,'lost',(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)

def takeScreenshot(img,box):
    pass
    
def searchImg(name,peopleInImg):
    if len(peopleInImg) > 0:
        for people in peopleInImg:
            if name == people[0]:
                return True
    return False

def createID(id):
    id_str = 'id' + str(id)
    id += 1
    return id,id_str

#function to draw tracking boxes
def drawBox(img,bbox,name,color='blue'):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    #cv2.rectangle(img,(x,y),((x+w),(y+h)),color=(255,255,0),thickness=3)
    if color == 'blue':
        cv2.rectangle(img,bbox,color=(255,255,0),thickness=3)
        cv2.putText(img,str(name),(bbox[0]+10,bbox[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
    elif color == 'red':
        y1,x2,y2,x1 = bbox
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),color=(0,0,255),thickness=3)
        cv2.putText(img,str(name),(x1+5,y2),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    #cv2.putText(img,'tracking',(75,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#function to store persons in file
def storeInFile(name):
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