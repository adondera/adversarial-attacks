import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

with open('imagenet_labels.txt') as f:
    labels = f.readlines()

model = resnet50(pretrained=True)
model.eval()
last_cnn_layer = [model.layer4[-1]]
image = Image.open('train/cat.6.jpg')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]
)

transform_without_normalize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]
)

cam = GradCAM(model=model, target_layers=last_cnn_layer, use_cuda=False)


tensor = transform(image)
input_batch = tensor.unsqueeze(0)
with torch.no_grad():
    output = model(input_batch)
probabilities = torch.softmax(output[0], dim=0)

top_probabilities = torch.topk(probabilities, 5)
for i, idx in enumerate(top_probabilities.indices):
    print(f'{labels[idx].rstrip()} - {top_probabilities.values[i] * 100 : .1f}%')

targets = [ClassifierOutputTarget(281)]
grayscale_cam = cam(input_tensor=input_batch, targets=targets)
grayscale_cam = grayscale_cam[0, :]
rgb_image = transform_without_normalize(image).transpose(0, 1).transpose(1,2).numpy()
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

img = Image.fromarray(visualization, 'RGB')
img.show()

