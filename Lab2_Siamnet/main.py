import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, data: np.ndarray, transform):
        """Initializes a dataset with a 4D array of images.

        Parameters
        ----------
        data : np.ndarray
            NumPy array with shape `(C,N,H,W)`, where
            * `C` is number of classes
            * `N` is number of images per class
            * `H`, `W` are height and width of images
        transform : _type_, optional
            `torch.nn` transform Module to be applied to images
        """
        super().__init__()

        self.data = data
        self.pairs = self.make_pairs(data)
        self.transform = transform

    def make_pairs(self, dataset: np.ndarray):
        num_classes = dataset.shape[0]
        num_images = dataset.shape[1]
        pairs = []

        for class_num, images in enumerate(dataset):
            for image_num, image in enumerate(images):

                positive_candidates = [i for i in range(num_images) if i != image_num]
                positive = images[random.choice(positive_candidates)]
                pairs.append((image, positive, 1.0))

                other_classes = [i for i in range(num_classes) if i != class_num]
                other_class = random.choice(other_classes)
                negative = dataset[other_class, random.randint(0, num_images-1)]
                pairs.append((image, negative, 0.0))

        return pairs

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        input_img1 = self.transform(img1)
        input_img2 = self.transform(img2)
        return input_img1, input_img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)


class SiamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32 * 28 * 23, 128)
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


def load_images_to_array(path: str, shape: tuple) -> np.ndarray:
    """ Loads a dataset into a 4D array of images
    to work with using indices.

    Parameters
    ----------
    path : str
        Path of a dataset directory
    shape : tuple
        Shape of the array: `(C,N,H,W)`, where
        * `C` is number of classes
        * `N` is number of images in one class
        * `H, W` are image width and height

    Returns
    -------
    np.ndarray
        Numpy array with images
    """
    data = np.zeros(shape, dtype=np.uint8)
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith('s'):
            class_idx = int(subdir[1:]) - 1
            for img_name in os.listdir(subdir_path):
                if img_name.endswith('.pgm'):
                    img_idx = int(img_name[:-4]) - 1
                    img_path = os.path.join(subdir_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    data[class_idx, img_idx] = img
    return data


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
    path = kagglehub.dataset_download(
        "kasikrit/att-database-of-faces",
        output_dir=os.path.join(root_dir, 'data')
    )

    # Image size: 112x92
    data = load_images_to_array(path, (40, 10, 112, 92))
    train_dataset = FaceDataset(data[0:35], train_transform)
    test_dataset = FaceDataset(data[35:40], test_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamNet().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(15):
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    figure = plt.figure(figsize=(6, 8))
    subfigures = figure.subfigures(ncols=2, nrows=5)
    subfig_iter = iter(subfigures.flat)
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            emb1, emb2 = model(img1, img2)
            distance = nn.functional.pairwise_distance(emb1, emb2).clamp(min=0.0, max=1.0)
            probability = 1.0 - float(distance)

            out_image1 = img1.squeeze().squeeze().cpu()
            out_image2 = img2.squeeze().squeeze().cpu()
            label = float(label.cpu())

            try:
                subfig = next(subfig_iter)
                subfig.suptitle(f'True: {label:.1f}, Eval: {probability:.2f}')
                axs = subfig.subplots(nrows=1, ncols=2)
                axs[0].imshow(out_image1, cmap='gray')
                axs[0].axis('off')
                axs[1].imshow(out_image2, cmap='gray')
                axs[1].axis('off')
            except StopIteration:
                plt.show()
                figure = plt.figure(figsize=(6, 8))
                subfigures = figure.subfigures(ncols=2, nrows=5)
                subfig_iter = iter(subfigures.flat)
        plt.show()


if __name__ == '__main__':
    main()
