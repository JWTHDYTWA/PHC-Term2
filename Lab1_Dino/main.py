import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
layer4_features = None
avgpool_emb = None

def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    model.layer4.register_forward_hook(get_features)
    model.avgpool.register_forward_hook(get_embedding)
    model.eval()



if __name__ == '__main__':
    main()
