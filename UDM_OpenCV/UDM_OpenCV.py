#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# ## Learning OpenCV with Python_An Udemy Course
# 0. [Let's Learn](#0)<br>
# 1. [Image Fundamentals](#1)<br>
#     1.1 [Introduction: What is an image](#11)<br>
#     1.2 [Mac OSX Python3 and Open CV installation](#12)<br>
#     1.3 [Windows Python and OpenCV installation](#13)<br>
#     1.4 [How to read an image](#14)<br>
#     1.5 [How to save an image in a different format](#15)<br>
# 2. [Drawing shapes and writing text on an image](#2)<br>
#     2.1 [Drawing](#21)<br>
#     2.2 [Drawing a rectangle](#22)<br>
#     2.3 [Drawing a line](#23)<br>
#     2.4 [Drawing a circle](#24)<br>
#     2.5 [Writing text](#25)<br>
#     2.6 [Drawing combination](#26)<br>
# 3. [Image Processing](#3)<br>
#     3.1 [Image Transformation](#31)<br>
#     3.2 [Image Rotation](#32)<br>
#     3.3 [Image Thresholding](#33)<br>  
# 4. [Image Filtering](#4)<br>
#     4.1 [Gaussian Blur](#41)<br>
#     4.2 [Median Blur](#42)<br>
#     4.3 [Bilateral Filtering](#43)<br>
# 5. [Feature Detection](#5)<br>
#     5.1 [Canny Edge Detector](#51)<br>
# 6. [Video Analysis](#6)<br>
#     6.1 [Load a video](#61)<br>
#     6.2 [Save a video in a different format](#62)<br>
# 7. [Applications](#7)<br>
#     7.1 [Introduction to Image Face Detection](#71)<br>
#     7.2 [Real time face detection using webcam](#72)<br>
#     7.3 [Image Face Detection](#73)<br>
#     1.4 [Real image face detection using webcam](#74)<br> 
# [go to bottom](#bottom)    

# # 0. Let's Learn <a id='0'></a>
# [go to top](#top)<br>

# - 

# # 1. Image Fundamentals <a id='1'></a>
# [go to top](#top)<br>

# ## 1.1 Introduction: What is an image <a id='11'></a>
# [go to top](#top)<br>

# ![image.png](attachment:image.png)

# ## 1.2 Mac OSX Python3 and Open CV installation <a id='12'></a>
# [go to top](#top)<br>

# In[1]:


#skip this one


# ## 1.3 Windows Python and OpenCV installation <a id='13'></a>
# [go to top](#top)<br>

# 1. Download anaconda https://www.continuum.io/downloads
# 2. download opencv poencv.org
# 3. put in c:\opencv\build\python\3.5.x\x64for 64 bit
# and put in c:\opencv\build\python\3.5.x\x68 for 32 bit
# 4. Find cv2.pyd copy to site packages in the anaconda directory
# this is site packages: c:\Users\user name\Anaconda\Lib\site-packages
# (why just use: conda install opencv ???)
# 5. Create an environment:
# variable: OPENCV_DIR Path c:\opencv\build\x86\vc12 for 32 bit
# c:\opencv\build\x64\vc12 for for 64 bit
# 
# 

# **my comments:**
# - this instruction is unnecessary because in anaconda we can use conda install opencv or conda install cv2.

# ## 1.4 How to read an image <a id='14'></a>
# [go to top](#top)<br>


# In[1]:


#Read an image with all color
import numpy as np
import cv2

img = cv2.imread('images//flower1.jpg')
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #It was not working,but after updating opencv, restart the jupyter notebook
# it working again

# In[2]:


#Read an image with grey color
import numpy as np
import cv2
img = cv2.imread('images//flower1.jpg',0)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#Read an image with all color
import numpy as np
import cv2
img = cv2.imread('images//flower1.jpg',-1)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## 1.5 How to save an image in a different format <a id='15'></a>
# [go to top](#top)<br>

# In[6]:


#Read an image with all color
# import numpy as np
import cv2
img1 = cv2.imread('images//flower1.jpg')
cv2.imwrite('images//flower2.png',img1)
img21 = cv2.imread('images//flower2.png',1)
cv2.imshow('Original',img21)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# In[23]:


get_ipython().run_line_magic('ls', 'images')


# # 2. Drawing shapes and writing text on an image <a id='2'></a>
# [go to top](#top)<br>

# ## 2.1 Drawing <a id='21'></a>
# [go to top](#top)<br>

# In[5]:


#1. Define the dimension of the image
#2. numpy zeros array
#3. implement functions


# ## 2.2 Drawing a rectangle <a id='22'></a>
# [go to top](#top)<br>

# In[8]:


#
import numpy as np
import cv2

pic = np.zeros((500,500,3),dtype = 'uint8') 
#cv2.rectangle(pic,(0,0),(500,150),(123,200,98),3,lineType=8,shift=0)
cv2.rectangle(pic,(0,0),(500,150),(0,0,250),3,lineType=8,shift=0)
cv2.rectangle(pic,(100,100),(250,150),(0,250,0),3,lineType=8,shift=0)
# color order is blue/green/red
cv2.imshow('dark',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# #### 2.3 Drawing a line <a id='23'></a>
# [go to top](#top)<br>

# In[11]:


#
import numpy as np
import cv2

pic = np.zeros((500,500,3),dtype = 'uint8')

cv2.line(pic,(350,350),(500,350),(250,0,0),4,lineType=8,shift=0) # color order is blue/green/red

cv2.imshow('dark',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## 2.4 Drawing a circle <a id='24'></a>
# [go to top](#top)<br>

# In[21]:


#
import numpy as np
import cv2

pic = np.zeros((500,500,3),dtype = 'uint8')

color1 = (255,100,255)
color2 = (255,100,0)
cv2.circle(pic,(250,250),80,color1,3,lineType=8)
cv2.circle(pic,(250,250),100,color2,3,lineType=8)
cv2.imshow('dark',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## 2.5 Writing text <a id='25'></a>
# [go to top](#top)<br>

# In[23]:


#
import numpy as np
import cv2

pic = np.zeros((500,1000,3),dtype = 'uint8')

font = cv2.FONT_HERSHEY_DUPLEX #can be found in documentation of opencv
cv2.putText(pic,'Python - OpenCV',(100,100),font,3,(255,255,255),4,cv2.LINE_8)


cv2.imshow('Image',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## 2.6 Drawing combination <a id='26'></a>
# [go to top](#top)<br>

# In[37]:


#
import numpy as np
import cv2
pic = np.zeros((500,800,3),dtype = 'uint8')
cv2.rectangle(pic,(0,0),(800,150),(123,200,98),8,lineType=8,shift=0)
cv2.rectangle(pic,(1,1),(800,150),(250,20,8),2,lineType=8,shift=0)
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(pic,'PyCheat & GIS',(50,100),font,3,(255,255,255),4,cv2.LINE_8)
cv2.circle(pic,(250,250),50,(255,0,255))
cv2.line(pic,(133,138),(388,133),(0,0,255))

cv2.imshow('Image',pic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# # 3. Image Processing <a id='3'></a>
# [go to top](#top)<br>

# # 3.1 Image Transformation <a id='31'></a>
# [go to top](#top)<br>

# In[5]:


ls images


# In[38]:


#Read an image with all color
import numpy as np
import cv2
img = cv2.imread('images//beckham1.jpg')
cv2.imshow('Original - Beckham',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[49]:


#shift the image
import numpy as np
import cv2

pic = cv2.imread('images//beckham2.jpg')
rows = pic.shape[0]
cols = pic.shape[1]

M = np.float32([[1,0,50],[0,1,10]]) #this matrix control how the image is shifted.

shifted = cv2.warpAffine(pic,M,(cols,rows))
cv2.imshow('shifted',shifted)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[43]:


M


# In[42]:


pic.shape


# In[25]:


print(M)
print(M[0,:])
print(M[1,:])


# In[50]:


import numpy as np
import cv2

pic = cv2.imread('images//beckham1.jpg')
rows = pic.shape[0]
cols = pic.shape[1]

M = np.float32([[1,0,-300],[0,1,-150]])

shifted = cv2.warpAffine(pic,M,(cols,rows))
cv2.imshow('shifted',shifted)

cv2.waitKey(0)
cv2.destroyAllWindows()



# In[9]:


pic


# # 3.2 Image Rotation <a id='32'></a>
# [go to top](#top)<br>

# In[51]:


import numpy as np
import cv2

pic = cv2.imread('images//rainbow_PNG5570.png')
rows = pic.shape[1]
cols = pic.shape[0]
center = (cols/2,rows/2)
angle = 90
M = cv2.getRotationMatrix2D(center,angle,0.5) #rotation matrix
rotate = cv2.warpAffine(pic,M,(cols,rows))
cv2.imshow('rotated',rotate)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[54]:


angle = [i for i in range(-180,190,10)]
angle


# In[56]:


import numpy as np
import cv2

pic = cv2.imread('images//beckham1.jpg')
rows = pic.shape[1]
cols = pic.shape[0]
center = (cols/2,rows/2)
angle_list = [i for i in range(-180,190,15)]
for angle in angle_list:
    M = cv2.getRotationMatrix2D(center,angle,0.5)
    rotate = cv2.warpAffine(pic,M,(cols,rows))
    cv2.imshow('rotated',rotate)

    cv2.waitKey(500)
    cv2.destroyAllWindows()


# In[60]:


import numpy as np
import cv2

pic = cv2.imread('images//horse.jpg')
rows = pic.shape[1]
cols = pic.shape[0]
center = (cols/2,rows/2)
angle_list = [i for i in range(-180,190,15)]
for angle in angle_list:
    M = cv2.getRotationMatrix2D(center,angle,0.75)
    rotate = cv2.warpAffine(pic,M,(cols,rows))
    cv2.imshow('rotated',rotate)

    cv2.waitKey(300)
    cv2.destroyAllWindows()


# # 3.3 Image Thresholding <a id='33'></a>
# [go to top](#top)<br>

# In[70]:


import cv2
import numpy as np

pic = cv2.imread('images//beckham1.jpg',1) #gray scale
threshold_value_list = [i for i in range(0,300,10)]
for threshold_value in threshold_value_list:
    (T_value,binary_threshold) = cv2.threshold(pic,threshold_value,255,cv2.THRESH_BINARY)
    cv2.imshow('binary',binary_threshold)
    cv2.waitKey(400)
    cv2.destroyAllWindows()


# In[64]:


import cv2
import numpy as np

pic = cv2.imread('images//rainbow1.jpg',0) #gray scale
threshold_value = 300

(T_value,binary_threshold) = cv2.threshold(pic,threshold_value,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary_threshold)
cv2.waitKey()
cv2.destroyAllWindows()


# In[17]:


import cv2
import numpy as np

pic = cv2.imread('images//rainbow1.jpg',0) #gray scale
threshold_value = 100

(T_value,binary_threshold) = cv2.threshold(pic,threshold_value,255,cv2.THRESH_BINARY_INV)
cv2.imshow('binary',binary_threshold)
cv2.waitKey()
cv2.destroyAllWindows()


# # 4. Image Filtering<a id='4'></a>
# [go to top](#top)<br>

# In[ ]:





# # 41. Gaussian Blur<a id='41'></a>
# [go to top](#top)<br>

# In[83]:


import cv2
import numpy
pic = cv2.imread('images//beckham1.jpg')
matrix = (7,7)

blur = cv2.GaussianBlur(pic,matrix,0)

cv2.imshow('blurred',blur)
cv2.waitKey()
cv2.destroyAllWindows()


# # 42. Median Blur<a id='42'></a>
# [go to top](#top)<br>

# In[84]:


#good to remove noise
import cv2
import numpy as np
pic = cv2.imread('images//horse.jpg')

kernal = 3

median = cv2.medianBlur(pic,kernal)

cv2.imshow('medium_blurred',median)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 43. Bilateral Filtering<a id='43'></a>
# [go to top](#top)<br>

# In[85]:


#how to make an image more smooth and clear.
import cv2
import numpy as np

pic = cv2.imread('images//horse2.png')

dimpixel = 7
color = 100
space = 100

filter = cv2.bilateralFilter(pic,dimpixel,color,space)
cv2.imshow('filter',filter)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 5. Feature Detection <a id='5'></a>
# [go to top](#top)<br>

# # 51. Canny Edge Detector <a id='51'></a>
# [go to top](#top)<br>

# In[32]:


import cv2
import numpy as np

pic = cv2.imread('images//dice_black1.jpg')

thresholdval1 = 50
thresholdval2 = 100

canny = cv2.Canny(pic,thresholdval1,thresholdval2)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[88]:


import cv2
import numpy as np

pic = cv2.imread('images//Dices.png')

thresholdval1 = 50
thresholdval2 = 100

canny = cv2.Canny(pic,thresholdval1,thresholdval2)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 6. Video Analysis<a id='6'></a>
# [go to top](#top)<br>

# # 61. Load a video<a id='61'></a>
# [go to top](#top)<br>

# In[93]:


import cv2
import numpy as np

cap = cv2.VideoCapture('videos//samplevideo.mp4')

while(cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('video imshow',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # 62. Save a video in a different format - notwork<a id='62'></a>
# [go to top](#top)<br>

# In[94]:


import cv2
import numpy as np

cap = cv2.VideoCapture('videos//samplevideo.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30 #frame per second
framesize = (720,480)
out = cv2.VideoWriter('videos//sample.avi',fourcc,fps,framesize)

while(cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[5]:


#Why I can't read the output video?
import cv2
import numpy as np

cap = cv2.VideoCapture('videos//sample.avi')

while(cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # 7. Applications <a id='7'></a>
# [go to top](#top)<br>

# # 71. Introduction to Image Face Detection <a id='71'></a>
# [go to top](#top)<br>

# In[ ]:


#use pre-trained model in open cv.


# # 72. Real time face detection using webcam <a id='72'></a>
# [go to top](#top)<br>

# In[7]:


face_cascade


# In[1]:


import cv2
import numpy as np
#https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('C://Dropbox//Study//Udemy_OpenCV//opencv-master//data//haarcascades_cuda//haarcascade_frontalface_default.xml')

videocapture = cv2.VideoCapture(0)
scale_factor = 1.3

while 1:
    ret, pic = videocapture.read()
    
    faces = face_cascade.detectMultiScale(pic,scale_factor,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(pic,'Me',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
        
    print("Number of faces found {} ".format(len(faces)))
    cv2.imshow('face',pic)
    k = cv2.waitKey(30)&0xff
    if k ==2:
        break
cv2.destroyAllWindows()


# # 73. Image Face Detection <a id='73'></a>
# [go to top](#top)<br>

# In[3]:


import cv2
import numpy as np
#https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('C://Dropbox//Study//Udemy_OpenCV//opencv-master//data//haarcascades_cuda//haarcascade_frontalface_default.xml')


pic = cv2.imread('images//beckham1.jpg')
scale_factor = 1.3

while 1:
    faces = face_cascade.detectMultiScale(pic,scale_factor,2)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),3)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(pic,'David Beckam',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
        
    print("Number of faces found {} ".format(len(faces)))
    cv2.imshow('face',pic)
    k = cv2.waitKey(30)&0xff
    if k ==2:
        break
cv2.destroyAllWindows()


# # 74. Real image face detection using webcam <a id='74'></a>
# [go to top](#top)<br>

# In[1]:


import cv2
import numpy as np
#https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('C://Dropbox//Study//Udemy_OpenCV//opencv-master//data//haarcascades_cuda//haarcascade_frontalface_default.xml')

videocapture = cv2.VideoCapture(0)
scale_factor = 1.3

while 1:
    ret, pic = videocapture.read()
    
    faces = face_cascade.detectMultiScale(pic,scale_factor,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(pic,'Me',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
        
    print("Number of faces found {} ".format(len(faces)))
    cv2.imshow('face',pic)
    k = cv2.waitKey(30)&0xff
    if k ==2:
        break
cv2.destroyAllWindows()


# 

# <a id='bottom'></a>
# [go to top](#top)<br>

# **Step to prepare a table of content for a new notebook**
# * Make a copy of this note book and rename to new notebook
# * change the table of content to fit with your table of content of the new notebook
# * copy **take home note* and  * ``- and change to scipts folder and run script in each session of your new notebook.
