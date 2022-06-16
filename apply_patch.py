import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import cv2

with open('imagenet_labels.txt') as f:
    labels = f.readlines()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.eval()
# Select an image to apply the patch to.
image = Image.open('train/cat.4.jpg')
patch = Image.open('best_patch.png')

fig, axes = plt.subplots(4, 2)
axes[0, 0].set_title("Original image")
axes[1, 0].set_title("Perturbed image")
axes[2, 0].set_title("Original image defended")
axes[3, 0].set_title("Perturbed image defended")
axes[0, 1].set_title("Top 3 probable classes")
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


def get_grayscale_cam(tensor):
    targets = None  # Target set to None to look at highest scoring class
    grayscale_cam = cam(input_tensor=normalize(tensor).unsqueeze(0), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    rgb_image = inv_normalize(tensor).permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization, 'RGB')
    img.show()
    return grayscale_cam


def create_masked_image(grayscale_tensor, image_tensor, threshold=0.9):
    img = cv2.cvtColor(grayscale_tensor, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    masked_image = (1 - torch.Tensor(thresh).permute(2, 0, 1)[:3]) * image_tensor
    return masked_image


last_cnn_layer = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=last_cnn_layer, use_cuda=False)
threshold = 0.9

image_tensor = convert_to_tensor(transform(image))
axes[0, 0].imshow(image_tensor.permute(1, 2, 0))

good_probabilities = get_topk_probabilities(image_tensor, 3)
plot_probabilities(good_probabilities[0], good_probabilities[1], axes[0, 1])

grayscale_cam = get_grayscale_cam(image_tensor)
masked_image = create_masked_image(grayscale_cam, image_tensor, threshold)
axes[2, 0].imshow(masked_image.permute(1, 2, 0))
fixed_probabilities = get_topk_probabilities(masked_image, 3)
plot_probabilities(fixed_probabilities[0], fixed_probabilities[1], axes[2, 1])

get_grayscale_cam(masked_image)

patch_tensor = convert_to_tensor(patch)[:3]
image_tensor[:, 0:patch_tensor.shape[1], 0:patch_tensor.shape[2]] = patch_tensor
axes[1, 0].imshow(image_tensor.permute(1, 2, 0))

adv_probabilities = get_topk_probabilities(image_tensor, 3)
plot_probabilities(adv_probabilities[0], adv_probabilities[1], axes[1, 1])

grayscale_cam = get_grayscale_cam(image_tensor)
masked_image = create_masked_image(grayscale_cam, image_tensor, threshold)
axes[3, 0].imshow(masked_image.permute(1, 2, 0))
fixed_probabilities = get_topk_probabilities(masked_image, 3)
plot_probabilities(fixed_probabilities[0], fixed_probabilities[1], axes[3, 1])

get_grayscale_cam(masked_image)

fig.tight_layout()

plt.show()
