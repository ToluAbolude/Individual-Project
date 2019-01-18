# import the necessary packages
from imutils import face_utils
from PIL import Image
from sys import argv
from os.path import exists
import numpy as np
import math
import argparse
import imutils
import dlib
import cv2
import glob
import cv2

videoFile = "test.mp4"
imagesFolder = "C:\Users\David Abolude\Desktop\Individual Project\img"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate (each 16.47 sec in the video)
#print "This is frame rate ",frameRate
while(cap.isOpened()):
	frameId = cap.get(1) #current frame number
	ret, frame = cap.read()
	if (ret != True):
		break
	if (frameId%math.floor(frameRate) == 0):
		filename = imagesFolder + "/image_" +  str(int(frameId)) + ".png"
		cv2.imwrite(filename, frame)
cap.release()
print "Done!"
for filename in glob.glob('./Img/*.png'): #assuming
	print filename
	
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(filename)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number (Text To Indicate drowsiness HERE!)
		cv2.putText(image, "Face {}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	cv2.waitKey(0)