import cv2
from copy import copy
import numpy as np

#Define VideoCapture object
cap=cv2.VideoCapture(1)

while True:
    #Get Video Frames from Camera and save a copy
    if cap:
        ret,capture=cap.read()
        video=capture.copy()

    #Video Processing Algorithm
    # Blurring, then Adaptive Thresholding, then Closing, Edge Detection
    #Thresholding, Contour Finding
    
    kernel=np.ones((1,1),np.uint8)
    img=cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    blur1=cv2.GaussianBlur(img,(7,7),0)
    blur2=cv2.morphologyEx(blur1,cv2.MORPH_CLOSE,kernel)
    th1=cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    process2=cv2.morphologyEx(th1,cv2.MORPH_CLOSE,kernel)
    edges=cv2.Canny(process2,50,150,apertureSize=3)
    ret,thresh=cv2.threshold(process2,127,255,0)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Grid Detection in Video
    
    maxArea=0
    val=0
    for i in range(len(contours)):
        area=cv2.contourArea(contours[i])
        if area>75000 and area>maxArea:
                maxArea=area
                val=i
    
    if maxArea>75000:
        x,y,w,h=cv2.boundingRect(contours[val])
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)
        grid=process2[y:y+h,x:x+w]
        grid1=cv2.morphologyEx(grid,cv2.MORPH_CLOSE,kernel,iterations=5)
    
    #Show the Final Images
    if maxArea>75000:
        cv2.imshow('The Grid',grid1)
    cv2.imshow('Video',video)
    #cv2.imshow('Edges',edges)
    k=cv2.waitKey(5) & 0xFF
    if k==ord('q'):
        break
    elif k==ord('s'):
        #Save the Image when 's' is clicked
        cv2.imwrite('savedsudoku.jpg',process2)
        cv2.imwrite('savedsudokucolor.jpg',capture)

cap.release()
cv2.destroyAllWindows()
