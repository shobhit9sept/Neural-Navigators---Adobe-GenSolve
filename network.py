import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import models, transforms

class SiameseNetworkDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        # Freeze the feature extractor layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1):
        output1 = self.feature_extractor(x1)
        return output1
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, label):
        euclidean_distance = torch.norm(output1 - label, p=2, dim=1)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive