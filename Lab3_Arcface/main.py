import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class SATDataset(Dataset):

    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform
        data = []
        class_labels = {}
        classes = 0
        for dir_name in os.listdir(path):
            dir_path = os.path.join(path, dir_name)
            if os.path.isdir(dir_path):
                class_labels[classes] = dir_name
                list_dir = os.listdir(dir_path)
                list_len = len(list_dir)
                for image_name in list_dir:
                    image_path = os.path.join(dir_path, image_name)
                    image = cv2.imread(image_path)
                    data.append((image, classes))
                classes += 1
        self.class_labels = class_labels
        self.data = data
    
    def __getitem__(self, index):
        return ...


class ArcFace(nn.Module):

    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            CosineComponent(64*64*32//4, 10)
        )
    
    def forward(self, x):
        return self.extractor(x)


class CosineComponent(nn.Module):
    
    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # Step 1:
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2:
        return x_norm @ W_norm


class ArcFaceLoss(nn.Module):

    def __init__(self, output_classes: int, m=0.4, s=64.0):
        super().__init__()
        self.output_classes = output_classes
        self.m = m
        self.s = s

    def forward(self, cosine: torch.Tensor, target):
        cosine = cosine.clip(-1+1e-7, 1-1e-7)
        arcosine = cosine.arccos()
        arcosine += F.one_hot(target, num_classes=self.output_classes) * self.m
        cosine2 = arcosine.cos()
        cosine2 = cosine2 * self.s
        return F.cross_entropy(cosine2, target)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


root_dir = os.path.dirname(__file__)


def main():
    batch_size = 16
    num_classes = 10

    train_dataset = SATDataset(os.path.join(root_dir,'data/split/train'), )
    
    model = ArcFace()
    criterion = ArcFaceLoss(output_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader()
    
    model.train()

    for epoch in range(69):
        for a,b,c in train_loader:
            ...
    
    # cosine_predictions = model(images) 
    # loss = criterion(cosine_predictions, targets)
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()


if __name__ == '__main__':
    main()