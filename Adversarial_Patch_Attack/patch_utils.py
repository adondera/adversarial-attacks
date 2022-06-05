"""
Credits: Code was adapted from https://github.com/A-LinCui/Adversarial_Patch_Attack
"""

import numpy as np
import torch


# Initialize the patch
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
        # TODO: Change this to torch instead and move to device
    else:
        raise NotImplementedError
    return patch


# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location = np.random.randint(low=0, high=image_size[1] - patch.shape[1] - 1)
        y_location = np.random.randint(low=0, high=image_size[2] - patch.shape[2] - 1)
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    else:
        raise NotImplementedError
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location


# Test the patch on dataset
# TODO: Check this function
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_actual_total, test_success = 0, 0
    for (image, label) in test_loader:
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image
        label = label
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0].item() != label.item() and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            changed_image = torch.mul(mask.type(torch.FloatTensor),
                                      applied_patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            changed_image = changed_image.cuda()
            output = model(changed_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total
