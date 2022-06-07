# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.
# Created by Junbo Zhao 2020/3/19
# Ref: https://github.com/A-LinCui/Adversarial_Patch_Attack/blob/master/utils.py

# Data used from https://github.com/megvii-research/FSSD_OoD_Detection/issues/1

import numpy as np
import torch
import torchvision
from torchvision.models import resnet50
from torchvision import datasets, transforms
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(dataset_name, train_size, test_size, data_dir, batch_size, device):
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if dataset_name == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=test_transforms)
    else:
        train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)

    index = np.arange(len(train_dataset))
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index),
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index),
                             shuffle=False)
    return train_loader, test_loader

# Test the model on clean dataset
def test(model, dataloader, device):
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            print(labels)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            print(predicted)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == '__main__':
    pass