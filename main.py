'''
Untargeted attacks are implemented.

Reference: https://github.com/LeMinhThong/blackbox-attack
'''

import torch
from torchvision.models import resnet50
from utils import dataloader

from setup_cifar10_model import CIFAR10
from attacks.zoo_attack import zoo_attack
from attacks.boundary_attack import boundary_attack
from attacks.blackbox_attack import opt_attack

def run_attack():

    use_cuda=True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Models
    model = resnet50(pretrained=True)

    # model = CIFAR10()
    # model.load_state_dict(torch.load('./models/cifar10_model.pt'))

    num_images = 1

    # Datasets
    train_loader, _ = dataloader(dataset_name='imagenet', train_size=2000, test_size=2000, 
                                data_dir='./data/imagenet_val', batch_size=1, device=device)

    # train_loader, _ = dataloader(dataset_name='cifar10', train_size=10, test_size=10, 
    #                             data_dir='./data', batch_size=1, device=device)

    # Type of attacks
    # adv = zoo_attack(model, train_loader, num_samples=num_images)
    # adv = boundary_attack(model, train_loader, num_samples=num_images)
    adv = opt_attack(model, train_loader, num_samples=num_images)

if __name__ == '__main__':
    run_attack()