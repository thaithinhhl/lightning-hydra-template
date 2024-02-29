import numpy as np
import matplotlib.pyplot as plt

from xml.dom import minidom

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import tqdm
import requests
import tarfile

class IBDataset(Dataset):
    def __init__(self, data_dir ="D:\\ibug_300W_large_face_landmark_dataset\\", data_url ="http:\\dlib.net\\files\\data\\ibug_300W_large_face_landmark_dataset.tar.gz", root: str = 'ibug_300W_large_face_landmark_dataset/'):
        super().__init__()
        
        if data_dir:
            self.data_dir = data_dir
        if data_url:
            self.data_url = data_url
        self.root = self.data_dir + root
        
        self.data = minidom.parse(self.root + 'labels_ibug_300W.xml' )
        self.data = self.data.getElementsByTagName("image")
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image_path = image.getAttribute('file')
        
        keypoints = []
        for point in image.getElementsByTagName('part'):
            keypoints.append((int(point.getAttribute("x")),int(point.getAttribute("y"))))
            
        bbox = image.getElementsByTagName("box")[0]
        x_min = int(bbox.getAttribute("left"))
        y_min = int(bbox.getAttribute("top"))
        width = int(bbox.getAttribute("width"))
        height = int(bbox.getAttribute("height"))
        x_max = x_min + width 
        y_max = y_min + height
        
        img = Image.open(self.root + image_path)
        img = img.crop((x_min,y_min,x_max,y_max))
        
        keypoints = np.array(keypoints) - np.array([x_min,y_min])

        
        return img, keypoints
        
    def __len__(self):
        return len(self.data)
        
    @staticmethod
    def plot_keypoints(img, keypoints):
        plt.imshow(img)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, marker='.', c='red')  # Vẽ các keypoints lên ảnh
        plt.show()
        
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
            IBDataset.download_url(dataset_url, tar_path)
            print("Extracting dataset")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=img_dir)
            os.remove(tar_path)
            print("hoan thanh")
        else:
            print("da ton tai dataset")
            
def main():
    data = IBDataset()
    image,keypoints = data[1]
    IBDataset.plot_keypoints(image,keypoints)

if __name__ == "__main__":
    main()