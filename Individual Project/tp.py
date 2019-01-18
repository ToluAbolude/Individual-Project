import distanceFormulaCalculator
import Paths
import globals
from globals import ClassifierFiles
import numpy as np
import cv2
import time
import dlib
import math
import eyeCoordinates
import mouthCoordinates
from globals import Threshold
from globals import yawnFolder
import os
import openface
VIDEO_PATHS = []


readVideo('v.avi')#test video of faces


def distanceBetweenMouth(c):
    m_60,m_61,m_62,m_63,m_64,m_65,m_66,m_67 = 0,0,0,0,0,0,0,0
    m_60 = c[59]
    m_61 = c[60]
    m_62 = c[61]
    m_63 = c[62]
    m_64 = c[63]
    m_65 = c[64]
    m_66 = c[65]
    m_67 = c[66]
    x1 = distanceFormulaCalculator.distanceFormula(m_61,m_67)
    x2 = distanceFormulaCalculator.distanceFormula(m_62,m_66)
    x3 = distanceFormulaCalculator.distanceFormula(m_63,m_65)   
    return ((x1+x2+x3)/3)



def mouthPoints():
    return [60,61,62,63,64,65,66,67]
	
def distanceRightEye(c):
    eR_36,eR_37,eR_38,eR_39,eR_40,eR_41 = 0,0,0,0,0,0
    eR_36 = c[35]
    eR_37 = c[36]
    eR_38 = c[37]
    eR_39 = c[38]
    eR_40 = c[39]
    eR_41 = c[40]
    x1 = distanceFormulaCalculator.distanceFormula(eR_37,eR_41)
    x2 = distanceFormulaCalculator.distanceFormula(eR_38,eR_40) 
    return ((x1+x2)/2)

def distanceLeftEye(c):
    eL_42,eL_43,eL_44,eL_45,eL_46,eL_47 = 0,0,0,0,0,0
    eL_42 = c[41]
    eL_43 = c[42]
    eL_44 = c[43]
    eL_45 = c[44]
    eL_46 = c[45]
    eL_47 = c[46]
    x1 = distanceFormulaCalculator.distanceFormula(eL_43,eL_47)
    x2 = distanceFormulaCalculator.distanceFormula(eL_44,eL_46) 
    return ((x1+x2)/2)



def eyePoints():
    return [36,37,38,39,40,41,42,43,44,45,46,47]
	


def readVideo(video):
    global no,yes
    video_capture = cv2.VideoCapture(video)
    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor(ClassifierFiles.shapePredicter) #Landmark identifier
    face_aligner = openface.AlignDlib(ClassifierFiles.shapePredicter)

    u = 0
    while True:
        ret, frame = video_capture.read()
        if frame != None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # clahe_image = clahe.apply(gray)

            detections = detector(frame, 1) #Detect the faces in the image

            for k,d in enumerate(detections): #For each detected face
                shape = predictor(frame, d) #Get coordinates
                vec = np.empty([68, 2], dtype = int)
                coor = []
                for i in range(1,68): #There are 68 landmark points on each face
                    #cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1)
                    coor.append([shape.part(i).x, shape.part(i).y])
                    vec[i][0] = shape.part(i).x
                    vec[i][1] = shape.part(i).y

                #RightEye and LeftEye coordinates
                rightEye = eyeCoordinates.distanceRightEye(coor)
                leftEye = eyeCoordinates.distanceLeftEye(coor)
                eyes = (rightEye + leftEye)/2

                #Mouth coordinates
                mouth = mouthCoordinates.distanceBetweenMouth(coor)

                print(eyes,mouth)
                #prints both eyes average distance
                #prints mouth distance
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

			
if __name__ == '__main__': 
    VIDEO_PATHS = Paths.videosPaths()
    init()