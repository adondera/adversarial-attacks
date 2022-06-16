import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

transform_without_normalize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]
)

with open('imagenet_labels.txt') as f:
    labels = f.readlines()

# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor

def get_preds_grad(model, inputs, dataset_name, device, image, correct_class=None, batch_size=25, return_cpu=True, attack=True):
    last_cnn_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=last_cnn_layer, use_cuda=False)

    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.to(device), volatile=True)
        with torch.no_grad():
            output = model(input_var)
        probabilities = softmax.forward(output)

        # targets = [ClassifierOutputTarget(674)]
      
        for j in range(3):
          top_probabilities = torch.topk(probabilities[j], 5)
          for i, idx in enumerate(top_probabilities.indices):
              print(f'{labels[idx].rstrip()} - {top_probabilities.values[i] * 100 : .1f}%')

          targets = [ClassifierOutputTarget(top_probabilities.indices[0])]
        
          grayscale_cam = cam(input_tensor=input_var, targets=targets)
          grayscale_cam = grayscale_cam[0, :]

          rgb_image = transform_without_normalize(image[j]).transpose(0, 1).transpose(1,2).numpy()
          visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

          img = Image.fromarray(visualization, 'RGB')
          if attack:
            path = "save/grad_cam_imgs/att_grad" + str(j) + ".jpg"
            img.save(path)
            path = "save/grad_cam_imgs/att" + str(j) + ".jpg"
            image[j].save(path)
          else:
            path = "save/grad_cam_imgs/org_grad" + str(j) + ".jpg"
            img.save(path)
            path = "save/grad_cam_imgs/org" + str(j) + ".jpg"
            image[j].save(path)
        # img.show()
        