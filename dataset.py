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

# Load CSV data
df = pd.read_csv('problems/problems/frag0.csv', header = None)

arc_points = {}
for arc_number in df[0].unique():
    points = df[df[0] == arc_number][[2, 3]].values
    arc_points[arc_number] = points


def load_and_preprocess_images(base_path):
    data = []
    labels = []
    class_names = os.listdir(base_path)
    
    for index, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            with Image.open(img_path) as img:
                # Convert image to grayscale
                img_gray = img.convert('L')
                # Resize image if necessary, e.g., to 28x28
                img_resized = img_gray.resize((28, 28))
                # Convert image to numpy array
                img_array = np.array(img_resized)
                # Normalize the image data to 0-1
                img_normalized = img_array / 255.0
                
                data.append(img_normalized)
                labels.append(index)
    
    return np.array(data), np.array(labels), class_names

# Specify the base path where your folders are stored
base_path = 'Data'

# Load and preprocess the images
data, labels, class_names = load_and_preprocess_images(base_path)

# Print some information to verify
# print(f'Loaded {len(data)} images.')
# print(f'Class names: {class_names}')

# # Generate all possible combinations of arcs and plot them
# def plot_arcs(arcs):
#     plt.figure()
#     for arc in arcs:
#         points = arc_points[arc]
#         plt.plot(points[:, 0], points[:, 1])
#     plt.axis('off')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     return Image.open(buf)


def plot_arcs(arcs):
    plt.figure()
    for arc in arcs:
        points = arc_points[arc]
        plt.plot(points[:, 0], points[:, 1])
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)  # Adjust dpi if necessary for resolution
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    # Convert to RGB to ensure no alpha channel is included
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        return image, idx  # Returning index as label for simplicity
    

# arc_numbers = list(arc_points.keys())
# all_combinations = []
# for r in range(1, len(arc_numbers) + 1):
#     all_combinations.extend(combinations(arc_numbers, r))

# # Create dataset
# dataset = ArcShapesDataset(all_combinations)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Example: Retrieve one batch and display
# for images, labels in dataloader:
#     plt.imshow(images[0].permute(1, 2, 0))
#     plt.title(f'Label: {labels.item()}')
#     plt.show()
#     break