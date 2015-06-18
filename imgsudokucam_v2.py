import cv2
from copy import copy
import numpy as np
from random import random as rnd

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

setallgrids=set()

cap=cv2.VideoCapture(1)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 500)

while True:
    ret,capture=cap.read()
    video=capture.copy()

    #find the squares#
    squares = []
    kernel=np.ones((1,1),np.uint8)
    kernel2=np.ones((3,3),np.uint8)
    img=cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    blur1=cv2.GaussianBlur(img,(3,3),0)
    blur1=cv2.morphologyEx(blur1,cv2.MORPH_CLOSE,kernel,iterations=10)
    cv2.imshow('blur1',blur1)
    th1=cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    process2=cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel)
    #process2=cv2.morphologyEx(process2,cv2.MORPH_OPEN,kernel2)
    for i in range(100):
        process2=cv2.morphologyEx(process2,cv2.MORPH_CLOSE,kernel)
    edges=cv2.Canny(process2,50,150,apertureSize=3)

    lines=cv2.HoughLines(edges,1,np.pi/180,200)
    if lines!=None:
        for rho,theta in lines[0]:
            a=np.cos(theta)
            b=np.sin(theta)
            x0=a*rho
            y0=b*rho
            x1=int(x0-1000*b)
            y1=int(y0+1000*a)
            x2=int(x0+1000*b)
            y2=int(y0-1000*a)
            cv2.line(video,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('Contoured',video)
    cv2.imshow('Processed Image',process2)

    k=cv2.waitKey(5) & 0xFF
    if k==ord('q'):
        break
    elif k==ord('s'):
        cv2.imwrite('savedsudoku.jpg',process2)
        cv2.imwrite('savedsudokucolor.jpg',capture)

cap.release()
cv2.destroyAllWindows()
