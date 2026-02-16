import os
import cv2
import random
# import torch
import kagglehub

class FaceDataset():
    def __init__(self, img_data, transform):
        super().__init__()
        self.crops = self.get_crops(img_data)
        self.transform = transform
    
    def get_crops(self, data):
        '''Load some img data'''
        return data

    def __getitem__(self, idx):
        img1, img2, label = self.crops[idx]
        input_img1 = self.transform(img1)
        input_img2 = self.transform(img2)
        return input_img1, input_img2, label
    
    def __len__(self):
        return len(self.crops)

def main():
    path = kagglehub.dataset_download("kasikrit/att-database-of-faces", output_dir="./data")
    
    print(os.path.join(path, os.listdir(path)[2]))


if __name__ == '__main__':
    main()