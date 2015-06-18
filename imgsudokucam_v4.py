import cv2
from copy import copy
import numpy as np
import mahotas
import time
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum,average
from scipy.spatial import distance as dist
import scipy
from math import sqrt
from sudoku6 import *
import pickle
import os
import serial
import dbn

#Define default VideoCapture object and Global Variables
cap=cv2.VideoCapture(1)
#ser=serial.Serial()
#ser.port='/dev/ttyACM0'
#ser.baudrate=9600
#ser.open()
#ser.write('1')
allnums=[]
#file1=open('data.pkl','wb+')
#file1.seek(0,os.SEEK_END)
#size=file1.tell()
#file1.seek(0)
'''if size>0:
     addit=pickle.loads(file1)
     for item in addit:
          allnums.append(item)
else:
     allnums=[]
file1.close()'''
predicted=[]
gridstr=list('0'*81)
loopit=True

def partOfGrid(x,y,w,h,rows,cols):
     onerow=int(rows/9)
     onecol=int(cols/9)
     m=int((x+w/2)/onecol)
     n=int((y+h/2)/onerow)
     return (m,n)
     #return (k+1,l+l/3+1)

def returnPredict(outline,value):
     r,c=outline.shape
     diffs1=[];diffs2=[];diffs3=[]; diffs4=[]; diffs5=[]
     for j in range(9):
          check=allnums[j][0].copy()
          check=cv2.resize(check,(c,r))
          check1=scipy.array(check).flatten()
          outline1=scipy.array(outline).flatten()
          diff=check1-outline1
          z_norm=norm(diff.ravel(),0)
          diffs1.append((dist.cityblock(outline1,check1),j))
          diffs2.append((dist.hamming(outline1,check1),j))
          diffs3.append((dist.braycurtis(outline1,check1),j))
          diffs4.append((dist.correlation(outline1,check1),j))
          #diffs5.append((z_norm,j))
          #cityblock,z_norm
     diffs1.sort();diffs2.sort();diffs3.sort();diffs4.sort();#diffs5.sort()
     a,b=diffs1[0]; c,d=diffs2[0]; e,f=diffs3[0];g,h=diffs4[0];#o,p=diffs5[0]
     possible=[allnums[b][1],allnums[d][1],allnums[f][1],allnums[h][1]]
     mval=0
     predict=0
     for k in range(1,10):
          if possible.count(k)>=mval:
               mval=possible.count(k)
               predict=k
     if allnums[b][1]!=allnums[d][1] or allnums[d][1]!=allnums[f][1] or allnums[f][1]!=allnums[h][1]: #or h!=p:
          print b+1,d+1,f+1,h+1,"for the image",i, "hence predicted: ",predict
          cv2.imshow('predict_this',outline)
          cv2.waitKey(0)
          cv2.destroyWindow('predict_this')
          return predict
     else:
          return predict

def findMaxArea(contours,minval=65000):
     maxArea=0
     val=0
     for i in range(len(contours)):
         area=cv2.contourArea(contours[i])
         if area>minval and area>maxArea:
                 x,y,w,h=cv2.boundingRect(contours[i])
                 maxArea=area
                 val=i
     return maxArea,val

def warped(orig,gridcont):
     pts=gridcont.reshape(4,2)
     rect=np.zeros((4,2),dtype="float32")
     s=pts.sum(axis=1)
     rect[0]=pts[np.argmin(s)]
     rect[2]=pts[np.argmax(s)]
     diff=np.diff(pts,axis=1)
     rect[1]=pts[np.argmin(diff)]
     rect[3]=pts[np.argmax(diff)]
     (tl, tr, br, bl) = rect
     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
 
     # ...and now for the height of our new image
     heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
     heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))
      
     # take the maximum of the width and height values to reach
     # our final dimensions
     maxWidth = max(int(widthA), int(widthB))
     maxHeight = max(int(heightA), int(heightB))
 
     # construct our destination points which will be used to
     # map the screen to a top-down, "birds eye" view
     dst = np.array([
             [0, 0],
             [maxWidth - 1, 0],
             [maxWidth - 1, maxHeight - 1],
             [0, maxHeight - 1]], dtype = "float32")
      
     # calculate the perspective transform matrix and warp
     # the perspective to grab the screen
     M = cv2.getPerspectiveTransform(rect, dst)
     warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
     return warp
          
def datasetNums():
     allnums=[]
     dataset=cv2.imread('dataset1.jpg',0)
     rows,cols=dataset.shape
     for i in range(3):
         for j in range(3):
             roi=dataset[i*(rows/3)+50:(i+1)*(rows/3)-50,j*(cols/3)+50:(j+1)*(cols/3)-50].copy()
             roi=255-roi
             allnums.append((roi,(i*3+j+1)))
             #cv2.imshow('image%s'%format(str(i)+str(j)),roi)
     return allnums

while loopit==True:
    #Get Video Frames from Camera and save a copy
    ret,capture=cap.read()
    if ret:
        video=capture.copy()
    else:
        video=cv2.imread('sudoku.png',1)
        loopit=False

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
    contours,hierarchy=cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if allnums==[]:
        allnums=datasetNums()
    
    #Grid Detection in Video
    maxArea,val=findMaxArea(contours)
    x,y,w,h=cv2.boundingRect(contours[val])
    grid1=None
    if maxArea>60000 and 0.7*maxArea<w**2<1.2*maxArea and 0.7*maxArea<h**2<1.2*maxArea:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)         #bounding rect for grid
        cv2.drawContours(video,[contours[val]],0,(0,0,255),2)    #red colour actual contour
        grid1=process2[y:y+h,x:x+w]
        gridcolor=video[y:y+h,x:x+w].copy()
        
        #process the grid
        r1,thresh1=cv2.threshold(grid1,127,255,0) 
        c,hier=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        imgarr=[]
        for i in range(len(c)):
            if 60<cv2.contourArea(c[i])<350:
                x,y,w,h=cv2.boundingRect(c[i])
                if 200>h>5 and 100>w>5 and 200<w*h<650 and 0.3<float(w)/float(h)<1.3:
                    cv2.rectangle(gridcolor,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
                    num=grid1[y-5:y+h+5,x-5:x+w+5]
                    rows,cols=grid1.shape
                    imgarr.append(num)
                    alpha,beta=partOfGrid(y,x,h,w,rows,cols)
                    predicted.append((alpha,beta))
                    
                    #cv2.imshow('num%s'%format(len(imgarr)-1),num)
    #Show the Final Images
    index=dict()
    if maxArea>60000 and grid1!=None:
        cv2.imshow('The Grid',grid1)
        cv2.imshow('ColorGrid',gridcolor)
    cv2.imshow('Video',video)
    time.sleep(0.01)
    #cv2.imshow('Edges',edges)
    k=cv2.waitKey(5) & 0xFF
    if k==ord('q'):
        break
    elif k==ord('s'):
        #Save the Image when 's' is clicked
        cv2.imwrite('savedsudoku.jpg',grid1)
        cv2.imwrite('savedsudokucolor.jpg',capture)
    elif k==ord('w'):
         #Found the numbers in the grid, so clean up and identify them
         for i in range(len(imgarr)):
              p,q= imgarr[i].shape
              test = cv2.resize(imgarr[i],None,fx=p/30.0,fy=q/30.0)
              p,q=test.shape
              test=cv2.resize(test,None,fx=28.0/p,fy=28.0/q)
              test=test.reshape((1,784)).astype("int0")
              value=dbn.dbn.predict(test)
              cv2.imshow('predicted',imgarr[i])
              print value
              cv2.waitKey(0)
              cv2.destroyWindow('predicted')
              '''if value>0:
                   a,b=predicted[i]
                   gridstr[9*a+b]=str(value)'''
         print(gridstr)

if loopit==False:
     cap.release()
     for i in range(len(imgarr)):
          p,q= imgarr[i].shape
          test = cv2.resize(imgarr[i],None,fx=p/30.0,fy=q/30.0)
          p,q=test.shape
          test=cv2.resize(test,None,fx=28.0/p,fy=28.0/q)
          test=test.reshape((1,784)).astype("int0")
          value=dbn.dbn.predict(test)
          cv2.imshow('predicted',imgarr[i])
          print value
          cv2.waitKey(0)
          cv2.destroyWindow('predicted')

          if value>0:
               a,b=predicted[i]
               gridstr[9*a+b]=str(value)
     gridstr=''.join(gridstr)
     print(gridstr)
'''
     solvable=17-(81-gridstr.count('0'))
     solve_str(gridstr)
     if complete(gridarray[1]):
          print "Solved the grid"
          print(solve_str(gridstr))
     elif error(gridarray[1]):
          print "Some error in solving...."
     #print 'Unsolvable right now...'
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
