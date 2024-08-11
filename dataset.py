import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from PIL import Image
import os
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


df = pd.read_csv('problems/problems/frag0.csv', header = None)

arc_points = {}
for arc_number in df[0].unique():
    points = df[df[0] == arc_number][[2, 3]].values
    arc_points[arc_number] = points


def load_and_preprocess_images(base_path):
    data = []
    labels = []
    class_names = []
    img_names = os.listdir(base_path)  
    
    for index, img_name in enumerate(img_names):
        img_path = os.path.join(base_path, img_name)
        if os.path.isfile(img_path):  
            with Image.open(img_path) as img:               
                img_gray = img.convert('L')               
                img_resized = img_gray.resize((100, 100))                
                img_array = np.array(img_resized)              
                img_normalized = img_array / 255.0
                data.append(img_normalized)
                labels.append(index)  
                class_names.append(img_name)  
    return np.array(data), np.array(labels), class_names


base_path = 'Data'

data, labels, class_names = load_and_preprocess_images(base_path)


def plot_arcs(arcs):
    plt.figure()
    for arc in arcs:
        points = arc_points[arc]
        plt.plot(points[:, 0], points[:, 1])
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100) 
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    image = image.convert('RGB')
    return image


class ArcShapesDataset(Dataset):
    def __init__(self, arc_combinations):
        self.arc_combinations = arc_combinations
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        self.images = [plot_arcs(comb) for comb in arc_combinations]
        self.all_images_tensor = torch.stack(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        return image, self.all_images_tensor, idx
