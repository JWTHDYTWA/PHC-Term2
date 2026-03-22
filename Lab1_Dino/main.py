import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os

from torchvision.models.resnet import ResNet18_Weights

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
layer4_features = None
avgpool_emb = None

lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

def extract_feat(img: np.ndarray):
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        model(t)
    return (layer4_features.clone(), avgpool_emb.clone())

def create_heatmap(obj_avgpool, img_layer4, image_w, image_h):
    heatmap_tensor = 1 - torch.nn.functional.cosine_similarity(obj_avgpool, img_layer4).squeeze(0)
    heatmap_min = heatmap_tensor.min()
    heatmap_max = heatmap_tensor.max()
    heatmap_norm = (heatmap_tensor - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap_np = heatmap_norm.numpy()

    heatmap = cv2.resize(heatmap_np, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

root_dir = os.path.dirname(__file__)


def main():
    image = cv2.imread(os.path.join(root_dir, 'img/abc.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample_dino = image[860:950,330:610].copy()
    sample_dino[40:,200:] = [255,255,255]
    mask = cv2.inRange(sample_dino, lower_white, upper_white)
    sample_dino[mask == 255] = [0, 0, 0]

    image = cv2.imread(os.path.join(root_dir, 'img/A_B_C.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]

    plt.imshow(sample_dino)
    plt.show()

    model.layer4.register_forward_hook(get_features)
    model.avgpool.register_forward_hook(get_embedding)
    model.eval()

    img_layer4, _ = extract_feat(image)
    _, dino_avgpool = extract_feat(sample_dino)

    heatmap = create_heatmap(dino_avgpool, img_layer4, image_w, image_h)
    overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

    plt.imshow(overlay)
    plt.savefig(os.path.join(root_dir, 'fig1.jpg'))
    plt.show()


if __name__ == '__main__':
    main()
