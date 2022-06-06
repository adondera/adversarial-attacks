import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from pytorch_grad_cam import GradCAM
import cv2
import random
import wandb

from Adversarial_Patch_Attack.utils import dataloader

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

with open('imagenet_labels.txt') as f:
    labels = f.readlines()

patch = Image.open('best_patch.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = models.resnet50(pretrained=True).to(device)
model.eval()

patch_tensor_original = convert_to_tensor(patch)[:3]

_, test_loader = dataloader(train_size=1, test_size=2000, batch_size=4, device=device, data_dir="imagenet_val")

target_label = 859

last_cnn_layer = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=last_cnn_layer, use_cuda=True)
threshold = 0.8

wandb.init(project="adv-attacks", entity="adondera", tags=['test'], config={
    'threshold': threshold
})

targets = None


def get_masks(gray_cam):
    masks = []
    for i in range(gray_cam.shape[0]):
        img = cv2.cvtColor(grayscale_cam[i], cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        mask = (1 - torch.Tensor(thresh).permute(2, 0, 1)[:3]).to(device)
        masks.append(mask)
    masks = torch.stack(masks)
    return masks


true_labels = []
pred_labels = []
defended_labels = []
perturbed_labels = []
defended_perturbed_labels = []
for inputs, labels in tqdm.tqdm(test_loader):
    unnormalized_inputs = inv_normalize(inputs)

    with torch.no_grad():
        outputs = model(inputs)
        pred_label = torch.max(outputs, dim=1)[1]

    grayscale_cam = cam(input_tensor=inputs, targets=targets)
    masks = get_masks(grayscale_cam)
    defended_basic_inputs = unnormalized_inputs * masks
    with torch.no_grad():
        defended_basic_outputs = model(normalize(defended_basic_inputs))
        defended_basic_labels = torch.max(defended_basic_outputs, dim=1)[1]

    rotation = np.random.randint(0, 4)
    patch_tensor = torch.rot90(patch_tensor_original, rotation, dims=[1, 2])
    x = np.random.randint(low=0, high=224 - patch_tensor.shape[1] - 1)
    y = np.random.randint(low=0, high=224 - patch_tensor.shape[2] - 1)
    unnormalized_inputs[:, :, x:x + patch_tensor.shape[1], y:y + patch_tensor.shape[2]] = patch_tensor
    normalized_perturbed_inputs = normalize(unnormalized_inputs)
    with torch.no_grad():
        perturbed_outputs = model(normalized_perturbed_inputs)
        perturbations = torch.max(perturbed_outputs, dim=1)[1]

    grayscale_cam = cam(input_tensor=normalized_perturbed_inputs, targets=targets)
    masks = get_masks(grayscale_cam)

    masked_images = unnormalized_inputs * masks
    normalized_masked_images = normalize(masked_images)

    with torch.no_grad():
        defense_outputs = model(normalized_masked_images)
        defense_labels = torch.max(defense_outputs, dim=1)[1]

    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(pred_label.cpu().numpy())
    defended_labels.extend(defended_basic_labels.cpu().numpy())
    perturbed_labels.extend(perturbations.cpu().numpy())
    defended_perturbed_labels.extend(defense_labels.cpu().numpy())

print(f"Accuracy before perturbations: {accuracy_score(true_labels, pred_labels)}")
print(f"Accuracy after perturbations: {accuracy_score(true_labels, perturbed_labels)}")
print(f"Accuracy of original images after defense: {accuracy_score(true_labels, defended_labels)}")
print(f"Accuracy of perturbed images after defense: {accuracy_score(true_labels, defended_perturbed_labels)}")
total_toasters = len([x for x in true_labels if x == target_label])


wandb.log({
    "Accuracy before perturbations": accuracy_score(true_labels, pred_labels),
    "Accuracy after perturbations": accuracy_score(true_labels, perturbed_labels),
    "Accuracy of original images after defense": accuracy_score(true_labels, defended_labels),
    "Accuracy of perturbed images after defense": accuracy_score(true_labels, defended_perturbed_labels),
    "Total number of toasters": total_toasters
})

cf_matrix = confusion_matrix(true_labels, pred_labels, labels=np.arange(0, 1000))
print(f"Objects classified as toasters before: {sum(cf_matrix[:, 859])}")
print(f"Target class with most classified objects has {max(np.sum(cf_matrix, axis=0))} samples")


cf_matrix_perturbed = confusion_matrix(true_labels, perturbed_labels, labels=np.arange(0, 1000))
sns.heatmap(cf_matrix_perturbed[:, 800:900])
print(f"Objects classified as toasters after: {sum(cf_matrix_perturbed[:, 859])}")
print(f"Target class with most classified objects has {max(np.sum(cf_matrix_perturbed, axis=0))} samples")

plt.show()
