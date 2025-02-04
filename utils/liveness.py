import cv2
import dlib
import numpy as np
from scipy.spatial import distance

class LivenessDetector:
    def __init__(self):
        self.eye_ar_thresh = 0.25  
        self.eye_ar_frames = 3     
        self.detector = dlib.get_frontal_face_detector()  
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_head_movement(self, landmarks):
        nose_tip = landmarks[30]

        if nose_tip[0] < 200:  
            return "Left"
        elif nose_tip[0] > 400: 
            return "Right"
        else:
            return "Center"

    def detect_mouth_movement(self, landmarks):
        mouth = landmarks[48:68]

        upper_lip = mouth[13]  
        lower_lip = mouth[19]  
        mouth_open = distance.euclidean(upper_lip, lower_lip)

        if mouth_open > 20:  
            return "Open"
        else:
            return "Closed"

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            left_eye = landmarks[36:42]  
            right_eye = landmarks[42:48]  

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            head_direction = self.detect_head_movement(landmarks)
            cv2.putText(frame, f"Head: {head_direction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            mouth_state = self.detect_mouth_movement(landmarks)
            cv2.putText(frame, f"Mouth: {mouth_state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear < self.eye_ar_thresh:
                cv2.putText(frame, "Blink Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, len(faces) > 0