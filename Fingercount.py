import cv2
import os
import Handtracking as htm

wCam,hCam =800,800
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(3, hCam)

tipid = [4,8,12,16,20]
myimg = os.listdir("fingerimg")
print(myimg)

overlap =[]

for impath in myimg:
    image = cv2.imread(f'{"fingerimg"}/{impath}')
    overlap.append(image)
print(len(overlap))  

detector = htm.handDetector(detectionCon = 0.75)

while True:
    sucess,img = cap.read()
    cv2.flip(img,1)
    img = detector.findHands(img)
    lmlist = detector.findpositions(img,draw = False)
    #print(lmlist)

    if len(lmlist) != 0:
        fingers = []
        if lmlist[tipid[0]][1] > lmlist[tipid[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)

        
        for id in range(1,5):

            if lmlist[tipid[id]][2] < lmlist[tipid[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalfinger = fingers.count(1)
        # print(totalfinger) 
        if totalfinger == 1:
            h,w,c = overlap[5].shape
            img[0:h, 0:w] = overlap[5]
        elif totalfinger ==2:
            h,w,c = overlap[4].shape
            img[0:h, 0:w] = overlap[4]
        elif totalfinger == 3:
            h,w,c = overlap[1].shape
            img[0:h, 0:w] = overlap[1]  
        elif totalfinger == 4:
            h,w,c = overlap[0].shape
            img[0:h, 0:w] = overlap[0]      
        elif totalfinger == 5:
            h,w,c = overlap[2].shape
            img[0:h, 0:w] = overlap[2]
        elif totalfinger == 0:
            h,w,c = overlap[3].shape
            img[0:h, 0:w] = overlap[3]        

    
    cv2.imshow("FingerCount",img)
    cv2.waitKey(1)