# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from os import system, name 
import numpy as np
import playsound
import warnings
import argparse
import imutils
import time
import dlib
import sys
import cv2
import version2


def detectYawning():
	detector,predictor,vs = version2.loading()
	time.sleep(1.0)
	# grab the indexes of the facial landmarks for the mouth.
	(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
	
	# grab the indexes of the other necessary parts.
	(rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
	(lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
	(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
	(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
	
	yawns = 0
	yawn_status = False
	# initialize the frame counter as well as a boolean used to
	# indicate if the alarm is going off
	COUNTER = 0
	ALARM_ON = False
	# loop over frames from the video stream
	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=600)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		
		prev_yawn_status = yawn_status

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			
			# extract the top and bottom lips coordinates, then use the
			# coordinates to compute the distance between them.
			mouth = shape[mStart:mEnd]
			#=========================================
			topLips = np.squeeze([mouth[2],mouth[3],mouth[4],mouth[13],mouth[14],mouth[15]])
			top_lip_mean = np.mean(topLips, axis=0)
			top_lip = int(top_lip_mean[1])
			bottomLips = np.squeeze([mouth[8],mouth[9],mouth[10],mouth[17],mouth[18],mouth[19]])
			bottom_lip_mean = np.mean(bottomLips, axis=0)
			bottom_lip = int(bottom_lip_mean[1])
			
			# extract the points in the left right eyebrow as well as the jaw, and use the
			# coordinates to compute the distance between them.
			leftEyebrow = shape[lbStart:lbEnd]
			rightEyebrow = shape[rbStart:rbEnd]
			jaw = shape[jStart:jEnd]
			nose = shape[nStart:nEnd]
			
			#======================================
			jawArea = np.squeeze([jaw[5],jaw[8],jaw[11]])
			righteyebrow = rightEyebrow[1]
			lipsBottom = mouth[9]
			nosePoint = nose[6]
			
			A = dist.euclidean(righteyebrow,jawArea[0])
			B = dist.euclidean(lipsBottom,jawArea[1])
			
			faceSize = A + (B-1)
			# compute the convex hull for the left and right eye, then
			# visualize each of the top and bottom lips
			mouthHull = cv2.convexHull(mouth)
			
			cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
			lip_distance = abs(top_lip - bottom_lip)/faceSize
					
			if lip_distance > 0.18: #changed
				yawn_status = True 
				
				cv2.putText(frame, "Driver is Yawning", (0,50), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
				

				output_text = "Yawn Count: " + str(yawns + 1)

				cv2.putText(frame, output_text, (250,400),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
				
			 
			else:
				COUNTER = 0
				ALARM_ON = False
				yawn_status = False
				
			if prev_yawn_status == True and yawn_status == False:
				yawns += 1

			# draw the computed eye aspect ratio on the frame to help
			# with debugging and setting the correct eye aspect ratio
			# thresholds and frame counters
			#cv2.putText(frame, "EAR: {:.2f}".format(ear), (350, 30),
				#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "LipDistance: {:.2f}".format(lip_distance), (350, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		
		# check to see if the mouth state is at the yawn 
		# threshold, and if so, increment the blink frame counter

		# show the frame
		cv2.imshow("Driver Yawning Detection Demo", frame)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop() #sys.exit
	version2.clear()
	version2.main()