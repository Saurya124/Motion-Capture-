import cv2
import mediapipe as mp 
import time
import multiprocessing
import sys
import numpy as np

# source = "D:/Downloads/batta.mp4"
source = 1 # 0 for internal webcam, 1 for external camera, or path to video
def detect_pose(queue):

    initial_time = 0
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(source)
    bol,vidframe = cap.read()
    if not bol:
        cap.release()
        return

    width = vidframe.shape[1]
    height = vidframe.shape[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
        results = pose.process(frame_rgb)
        Current_time = time.time()
        frame_rate = 1/(Current_time-initial_time) if initial_time !=0 else 0
        initial_time = Current_time
        cv2.putText(frame, f'FPS: {int(frame_rate)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,results.pose_landmarks,
                [(11,12),(12,24),
                (24,23),(23,11),
                (11,13),(13,15),
                (12,14),(14,16),
                (24,26),(26,28),
                (23,25),(25,27),
                (0,8),(0,7)]
                )

            landmarks = [
                results.pose_landmarks.landmark[0],
                results.pose_landmarks.landmark[7],
                results.pose_landmarks.landmark[8],
                results.pose_landmarks.landmark[11],
                results.pose_landmarks.landmark[12],
                results.pose_landmarks.landmark[13],
                results.pose_landmarks.landmark[14],
                results.pose_landmarks.landmark[15],
                results.pose_landmarks.landmark[16],
                results.pose_landmarks.landmark[23],
                results.pose_landmarks.landmark[24],
                results.pose_landmarks.landmark[25],
                results.pose_landmarks.landmark[26],
                results.pose_landmarks.landmark[27],
                results.pose_landmarks.landmark[28],
                results.pose_landmarks.landmark[29],
                results.pose_landmarks.landmark[30],
                results.pose_landmarks.landmark[31],
                results.pose_landmarks.landmark[32]
            ]

            queue.put((landmarks,width,height))

        w = width
        h = height
        resize = cv2.resize(frame,(w,h))
        cv2.imshow('MediaPipe Pose', resize)
        time.sleep(1/60)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    queue.put("STOP")


