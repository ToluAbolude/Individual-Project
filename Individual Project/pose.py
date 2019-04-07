import cv2
import dlib
import imutils
import time
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import warnings
import numpy as np
import playsound
import argparse
import concurrent.futures
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imutils import face_utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
#ap.add_argument("-a", "--alarm", type=str, default="",
	#help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
x, y = pkl.load(open('./samples.pkl', 'rb'))

#print(x.shape, y.shape)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	
roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

#print(roll.min(), roll.max(), roll.mean(), roll.std())
#print(pitch.min(), pitch.max(), pitch.mean(), pitch.std())
#print(yaw.min(), yaw.max(), yaw.mean(), yaw.std())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

#print(x_train.shape, y_train.shape)
#print(x_val.shape, y_val.shape)
#print(x_test.shape, y_test.shape)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)

#Hyperparams
BATCH_SIZE = 64
EPOCHS = 100

# Training
#model = Sequential()
#model.add(Dense(units=20, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
#model.add(Dense(units=10, activation='relu', kernel_regularizer='l2'))
#model.add(Dense(units=3, activation='linear'))

#print(model.summary())

#callback_list = [EarlyStopping(monitor='val_loss', patience=25)]

#model.compile(optimizer='adam', loss='mean_squared_error')
#hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list)
#model.save('./model.h5')

#print()
#print('Train loss:', model.evaluate(x_train, y_train, verbose=2))
#print('  Val loss:', model.evaluate(x_val, y_val, verbose=2))
#print(' Test loss:', model.evaluate(x_test, y_test, verbose=2))

# Testing the model
def compute_features(face_points):
	#assert (len(face_points) == 68), "len(face_points) must be 68"
	features = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		for i in range(68):
			for j in range(i+1, 68):
				#print ("I: ",i,"J: ",j)
				features.append(np.linalg.norm(face_points[i]-face_points[j]))
		return(np.array(features).reshape(1, -1))

	
#im = cv2.imread('C:\\Users\\David Abolude\\Documents\\GitHub\\Individual-Project\\Individual Project\\headPose.jpg', cv2.IMREAD_COLOR)

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale channels)
	im = vs.read()
	im = imutils.resize(im, width=500)
	
	
	#-------- Improvement --------------
	rects = detector(im, 0)
	# loop over the face detections
	with concurrent.futures.ProcessPoolExecutor() as executor:
		for rect in rects:
			face_points = predictor(im,rect)
			face_points = face_utils.shape_to_np(face_points)
			with concurrent.futures.ProcessPoolExecutor() as executor:
				for x, y in face_points:
					cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
					
				features = compute_features(face_points)
				features = std.transform(features)

				model = load_model('./model.h5')
				y_pred = model.predict(features)
				print(y_pred)
				
				roll_pred, pitch_pred, yaw_pred = y_pred[0]
				cv2.putText(im, "Roll: {:.2f}".format(roll_pred), (300, 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(im, "Pitch: {:.2f}".format(pitch_pred), (300, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(im, "Yaw: {:.2f}".format(yaw_pred), (300, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
		cv2.imshow("Frame", im)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			# do a bit of cleanup
			cv2.destroyAllWindows()
			break


    
