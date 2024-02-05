import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy 
import numpy as np
from mtcnn.mtcnn import MTCNN

path_dir = "D:\\ibug_300W_large_face_landmark_dataset\\ibug_300W_large_face_landmark_dataset\\ibug" #duong dan file

list_files = (file for file in os.listdir(path_dir) if file.endswith(".jpg"))
list_files = list(list_files)
image_path = os.path.join(path_dir, list_files[0]) #duong dan den file dau tien

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#detector
detector = MTCNN()
faces = detector.detect_faces(image_rgb)

#crop image
if faces:
    x,y,h,w = faces[0]['box']
    face_crop = image_rgb[y:y+h,x:x+w]
    
    keypoints = [
        (336.820955, 240.864510),
        (334.238298, 260.922709),
        (335.266918, 283.697151),
        (339.307573, 302.270092),
        (344.609474, 321.426167),
        (350.930559, 340.781503)
    ]

    keypoints_crop = [(kp[0] - x, kp[1] - y) for kp in keypoints if x <= kp[0] <= x+w and y <= kp[1] <= y+h]

    for point in keypoints_crop:
        a ,b = point
        plt.scatter(a, b, color = 'red', s = 4)
        
    plt.imshow(face_crop)
    plt.axis('off')
    plt.show()
else:
    print("khong ton tai anh")