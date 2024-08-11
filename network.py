import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import models, transforms

class SiameseNetworkDataset(Dataset):
    def __init__(self, data, labels, class_images):
        self.data = data
        self.labels = labels
        self.class_images = class_images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img)
        label = self.labels[index]
        return img, self.class_images, label
    
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

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, label):
        euclidean_distance = torch.norm(output1 - label, p=2, dim=1)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    

def cosine_similarity(feature_vector, class_vectors):
    normalized_feature_vector = feature_vector / feature_vector.norm(dim=1, keepdim=True)
    normalized_class_vectors = class_vectors / class_vectors.norm(dim=1, keepdim=True)
    similarity = torch.mm(normalized_feature_vector, normalized_class_vectors.transpose(0, 1))
    return similarity
