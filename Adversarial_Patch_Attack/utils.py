# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.
# Created by Junbo Zhao 2020/3/19

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(train_size, test_size, data_dir, batch_size, device, total_num=50000):
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

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    # wandb.run.summary['train_index'] = train_index
    # wandb.run.summary['test_index'] = test_index
    #
    # np.save("train_index.npy", train_index)
    # np.save("test_index.npy", test_index)

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index),
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index),
                             shuffle=False)
    return train_loader, test_loader


# Test the model on clean dataset
def test(model, dataloader):
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in tqdm.tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total
