import cv2
from copy import copy
import numpy as np
from random import random as rnd

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

setallgrids=set()

cap=cv2.VideoCapture(1)
while True:
    ret,capture=cap.read()
    video=capture.copy()

    #find the squares#
    squares = []
    #img=cv2.imread('/home/rohan/Desktop/sudoku.jpg',0)
    #rows,cols,ch=img.shape
    kernel=np.ones((1,1),np.uint8)
    img=cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    #cv2.fastNlMeansDenoising(img,img,10,7,21) #too slow for real time use
    #orb=cv2.ORB()
    #kp=orb.detect(img,None)
    #kp,des=orb.compute(img,kp)

    #img2=cv2.drawKeypoints(img,kp,color=(0,255,0),flags=0)
    
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos <0.2:
                        squares.append(cnt)
    
    areas=[]
    index=0
    if len(squares)>1:
        for i in range(len(squares)):
            areas.append(cv2.contourArea(squares[i]))
        m=max(areas)
        index=areas.index(m)
       
    #corners=cv2.goodFeaturesToTrack(img,25,0.01,10)
    #corners=np.int0(corners)
    #dst=cv2.cornerHarris(gray,2,3,0.04)
    #dst=cv2.dilate(dst,None)
    #---Corners----# video[dst>0.01*dst.max()]=[0,0,255]
    blur1=cv2.GaussianBlur(img,(7,7),0)
    blur2=cv2.morphologyEx(blur1,cv2.MORPH_CLOSE,kernel)
    r,otsu=cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th1=cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    th2=cv2.adaptiveThreshold(blur2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

    process1=cv2.dilate(th2,kernel,iterations=1)
    process2=cv2.morphologyEx(th1,cv2.MORPH_CLOSE,kernel)
    edges=cv2.Canny(process2,50,150,apertureSize=3)
    #cv2.imshow('Original Image',video)

    #ret,thresh=cv2.threshold(edges,127,255,0)
    #contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    '''
    if len(squares)>1:
        cv2.drawContours(video,squares,-1,(0,255,0),3)
        print squares[0][1]
    #x,y,w,h=cv2.boundingRect(cnt)
    #cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)
    for i in corners:
        x,y=i.ravel()
        cv2.circle(video,(x,y),3,255,-1)

    '''
    minLineLength=100
    maxLineGap=10
    #---HOUGH PROBABILISTIC---#
    lines=cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines!=None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(video,(x1,y1),(x2,y2),(0,255,0),2)
    ##HOUGH PROBABILISTIC###
    '''
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
    '''
    cv2.drawContours(video,squares,index,(0,0,255),3)
    cv2.imshow('Contoured',video)
    cv2.imshow('Processed Image',process2)
    cv2.imshow('Edges',edges)
    k=cv2.waitKey(5) & 0xFF
    if k==ord('q'):
        break
    elif k==ord('s'):
        cv2.imwrite('savedsudoku.jpg',process2)
        cv2.imwrite('savedsudokucolor.jpg',capture)

cap.release()
cv2.destroyAllWindows()
