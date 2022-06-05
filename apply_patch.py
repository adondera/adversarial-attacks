import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch

with open('imagenet_labels.txt') as f:
    labels = f.readlines()

model = models.resnet50(pretrained=True)
model.eval()
image = Image.open('train/cat.4.jpg')
patch = Image.open('best_patch.png')

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(18.5, 10.5)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]
)

convert_to_tensor = transforms.ToTensor()


def get_topk_probabilities(im_tensor: torch.Tensor, k=10):
    normalized_tensor = normalize(im_tensor)
    input_batch = normalized_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.softmax(output[0], dim=0)
    top_probabilities = torch.topk(probabilities, k)
    return top_probabilities


def plot_probabilities(probabilities, indices, ax):
    classes = [labels[x] for x in indices]
    probabilities = [x * 100 for x in probabilities]
    ax.bar(classes, probabilities)


last_cnn_layer = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=last_cnn_layer, use_cuda=False)

image_tensor = convert_to_tensor(image)
axes[0, 0].imshow(image_tensor.permute(1, 2, 0))

good_probabilities = get_topk_probabilities(image_tensor, 3)
plot_probabilities(good_probabilities[0], good_probabilities[1], axes[0, 1])

patch_tensor = convert_to_tensor(patch)[:3]
image_tensor[:, 0:patch_tensor.shape[1], 0:patch_tensor.shape[2]] = patch_tensor
axes[1, 0].imshow(image_tensor.permute(1, 2, 0))

adv_probabilities = get_topk_probabilities(image_tensor, 3)
plot_probabilities(adv_probabilities[0], adv_probabilities[1], axes[1, 1])

plt.show()

targets = [ClassifierOutputTarget(859)]
grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)
grayscale_cam = grayscale_cam[0, :]
rgb_image = inv_normalize(image_tensor).permute(1, 2, 0).numpy()
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
img = Image.fromarray(visualization, 'RGB')
img.show()
