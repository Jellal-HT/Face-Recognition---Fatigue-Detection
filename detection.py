import cv2
import numpy as np
import dlib
import argparse
import imutils
import time
from collections import OrderedDict
from scipy.spatial import distance as dist

# function used to calculate the Eye Aspect Ratio，EAR
def calculateEAR(eye):
    # calculate verticle distance
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # claculate horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # EAR value
    EAR = (A + B) / (2.0 * C)
    return EAR
    
# function used to calculate the Mouth Aspect Ratio，MAR
def calculateMar(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    # MAR value
    mar = (A + B) / (2.0 * C)
    return mar
    
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
    
# set up the constants that are used to judge
MAR_THRESH = 0.5 # the threshold of MAR
MOUTH_AR_CONSEC_FRAMES = 3 # Yawn consecutive frames
EYE_AR_THRESH = 0.2 # the threshold of EAR
EYE_AR_CONSEC_FRAMES = 3 # Blink consecutive frames

# Initialize the frame counter for blink and the total number of blinks
COUNTER = 0
TOTAL = 0
# Initialize the frame counter for yawn and the total number of yawn
mCOUNTER = 0
mTOTAL = 0

# initialize the face detector(HOG) of DLIB, and establish the face landmark predictors
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

# get the region of two eyes and mouth
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = FACIAL_LANDMARKS_68_IDXS["mouth"]

# use cv2 to open the local camera
cap = cv2.VideoCapture(0)

# Loop frames from the video stream
while True:
    # read the picture from frames, expand the dimension of the picture, and grayscale the picture
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use detector(gray, 0) to detect the facial region
    rects = detector(gray, 0)
    
    # looping the facial region information
    for rect in rects:
        # translate the information into array format
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # get the left eye and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # get the mouth coordinates
        mouth = shape[mStart:mEnd]
        
        # calculate the EAR values of the left and right eyes
        leftEAR = calculateEAR(leftEye)
        rightEAR = calculateEAR(rightEye)
        # use the average as the final EAR value
        ear = (leftEAR + rightEAR) / 2.0
        # calculate the MAR value of the mouth
        mar = calculateMAR(mouth)
        
        # Use a rectangular box to mark the face
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3) 
        
        # use the threshold value to determine whether the object is yawning or not
        if mar > MAR_THRESH:# threshold value of MAR
            mCOUNTER += 1
            cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If it is less than the threshold 3 times in a row, it means that yawning only once
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值：3
                mTOTAL += 1
            # reset the counter
            mCOUNTER = 0
        # use the function(putText) in opencv to show the related result
        cv2.putText(frame, "mCOUNTER: {}".format(mCOUNTER), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # If the conditions are met, the number of blinks +1
        if ear < EYE_AR_THRESH:# 眼睛长宽比：0.2
            COUNTER += 1
           
        else:
            # If it is less than the threshold 3 times in a row, it means that yawning only once
            if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：3
                TOTAL += 1
            COUNTER = 0
        # use the function(putText) in opencv to show the related result
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    # determine the fatigue
    if TOTAL >= 50 or mTOTAL>=15:
        cv2.putText(frame, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    # press 'q' to exit
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # show with opencv using window
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# release the local camera and cleanup
cap.release()
cv2.destroyAllWindows()
