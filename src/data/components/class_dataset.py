
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import requests
import cv2
import matplotlib.patches as patches
import zipfile
from tqdm import tqdm
import tarfile

class CustomDataset(Dataset):
    def __init__(self, img_dir, dataset_url=None, transform=None, target_transform=None):
        if dataset_url is not None:
            self.prepare_data(img_dir, dataset_url)
        
        valid_extensions = ['.jpg', '.jpeg', '.png']
        self.img_names = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_extensions and os.path.isfile(os.path.join(img_dir, f))]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Không thể đọc hình ảnh từ {img_path}")
        
        label = 0  
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    @staticmethod
    def download_url(url, save_path):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(desc=save_path, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    @staticmethod
    def prepare_data(img_dir, dataset_url):
        tar_path = os.path.join(img_dir, "dataset.tar.gz")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)
            print(f"dang tai dataset {dataset_url}...")
            CustomDataset.download_url(dataset_url, tar_path)
            print("Extracting dataset")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=img_dir)
            os.remove(tar_path)
            print("hoan thanh")
        else:
            print("da ton tai dataset")

dataset_url = 'http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz'
img_dir = "D:\\ibug_300W_large_face_landmark_dataset\\ibug_300W_large_face_landmark_dataset\\ibug"


dataset = CustomDataset(img_dir)
image, label = dataset[0]

keypoints = [
 
    (336.820955, 240.864510),
    (334.238298, 260.922709),
    (335.266918, 283.697151),
    (339.307573, 302.270092),
    (344.609474, 321.426167),
    (350.930559, 340.781503)

]

for point in keypoints:
    x, y = point
    plt.scatter(x, y, color='green', s=5)  


image_grb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_grb)
plt.axis("off")
plt.show()

