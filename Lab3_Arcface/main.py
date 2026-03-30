import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SATDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = []
        class_labels = {}
        classes = 0
        for dir_name in os.listdir(path):
            dir_path = os.path.join(path, dir_name)
            if os.path.isdir(dir_path):
                if not (dir_name in class_labels):
                    class_labels[dir_name] = classes
                    classes += 1
                for image_name in os.listdir(dir_path):
                    image_path = os.path.join(dir_path, image_name)
                    image = cv2.imread(image_path)
    
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


root_dir = os.path.dirname(__file__)


def main():
    batch_size = 16
    num_classes = 10
    
    model = ArcFace()
    criterion = ArcFaceLoss(output_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    ...
    
    # cosine_predictions = model(images) 
    # loss = criterion(cosine_predictions, targets)
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()


if __name__ == '__main__':
    main()