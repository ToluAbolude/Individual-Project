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

#======================================
import head_pose
import facial_landmarks
import eye_closure
import yawning
#======================================

warnings.simplefilter(action='ignore', category=FutureWarning)


#================ Preprocess ===================

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

#===================================================


def clear():
	# for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

		
def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

	
def loading():
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	vs = VideoStream(src=args["webcam"]).start()
	return detector,predictor,vs

	
def main():
	print()
	print("    ************MAIN MENU**************")
	time.sleep(0.5)
	print()

	choice = input("""
	1: View Facial Landmarks.
	2: Detect Eye Closure (Using EAR)
	3: Detect Yawning.
	4: Detect Head Pose.
	X: Quit/Exit

	Please enter your choice: """)
	print()

	if choice == "1":
		facial_landmarks.displayFacialLandmarks()
	elif choice == "2":
		eye_closure.detectEar()
	elif choice == "3":
		yawning.detectYawning()
	elif choice== "4":
		head_pose.main()
	elif choice=="X" or choice=="x":
		sys.exit
	else:
		print("You must only select either 1,2,3, or 4.")
		print("Please try again")
		main()		

if __name__ == "__main__":
    # execute only if run as a script
    main()
