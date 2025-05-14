from detection import detect_pose
import multiprocessing
from math import sqrt

import os
import socket
import time
from queue import Empty
from filterpy.kalman import KalmanFilter
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = ('127.0.0.1', 5005)

def create_kf_3d():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1.0
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    kf.P *= 500
    kf.R = np.eye(3) * 0.1
    kf.Q = np.eye(6) * 0.01
    return kf

kf_dict = {i: create_kf_3d() for i in range(19)}

def process_landmarks(queue):
    
    open("land.txt", "w").close()

    while True:
        try:
            data = queue.get_nowait()
        except Empty:
            time.sleep(1/60) 
            continue

        if data == "STOP":
            break
        
        landmarks, width, height = data

        ref_distance = sqrt((landmarks[9].x * width - landmarks[10].x * width) ** 2 + 
                            (landmarks[9].y * height - landmarks[10].y * height) ** 2)
        zs = ref_distance * 2

        values = []
        data_send = []

        for i,lm in enumerate(landmarks):
            x = lm.x * width
            y = height - lm.y * height  # Flip Y for Blender
            z = lm.z * zs

            z_vec = np.array([x, y, z]).reshape((3, 1))

            
            kf = kf_dict[i]
            kf.predict()
            kf.update(z_vec)
            x_s, y_s, z_s = kf.x[:3].flatten()
           


            values.append(f"{x_s},{y_s},{z_s}")
            data_send.extend([x_s, y_s, z_s])  

        msg = ','.join(map(str, data_send)).encode('utf-8')
        sock.sendto(msg, server)

        with open("land.txt", "a") as file:
            file.write(",".join(values) + "\n")

        time.sleep(1/60)




if __name__ == "__main__":

    queue = multiprocessing.Queue()
    detect_process = multiprocessing.Process(target=detect_pose, args=(queue,))
    detect_process.start()
    process_landmarks(queue)
    detect_process.join()
