import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub

from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        images = []
        class_labels = []
        for subdir in os.listdir(path):
            subdir_path = os.path.join(path, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith('s'):
                class_label = int(subdir[1:])
                # print(class_label)
                for img_name in os.listdir(subdir_path):
                    if img_name.endswith('.pgm'):
                        img_path = os.path.join(subdir_path, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        images.append(img)
                        class_labels.append(class_label)

        self.pairs = self.make_pairs(images, class_labels)
        self.transform = transform
    
    def make_pairs(self, images, class_labels):
        images_np = np.array(images)
        labels_np = np.array(class_labels)
        labels_unique = np.unique(labels_np)
        pairs = []

        for class_label in labels_unique:

            for image in images_np[labels_np == class_label]:
                positive = ...

                pairs.append((image, positive, 1.0))
                ...
            ...
        print(labels_unique)

    def __getitem__(self, idx):
        img1, img2, label = self.crops[idx]
        input_img1 = self.transform(img1)
        input_img2 = self.transform(img2)
        return input_img1, input_img2, torch.tensor(label,dtype=torch.float32)
    
    def __len__(self):
        return len(self.crops)

class SiamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 110 * 90, 128)
        )

    def forward(self, img1, img2):
        return self.feature_extractor(img1), self.feature_extractor(img2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = torch.nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean(
            label * torch.pow(dist, 2) / 2 +
            (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2) / 2)
        return loss


root_dir = os.path.dirname(__file__)


def main():

    path = kagglehub.dataset_download(
        "kasikrit/att-database-of-faces",
        output_dir=os.path.join(root_dir, 'data')
    )
    
    dataset = D=FaceDataset(path, None)
    # Image size: 112x92
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SiamNet().to(device)
    # criterion = ContrastiveLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()