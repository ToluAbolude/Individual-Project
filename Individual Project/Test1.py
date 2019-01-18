# import the necessary packages
from imutils import face_utils
from PIL import Image
from sys import argv
from os.path import exists
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import cv2
image_list = []
#for filename in glob.glob('./Images/*.jpg'): #assuming 
	#im=Image.open(filename)
#im.show()
 
vidcap = cv2.VideoCapture('test.mp4')
success,image = vidcap.read()
count = 0
while success:
	vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
	cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
	success,image = vidcap.read()
	print('Read a new frame: ', success)
	count += 1 
