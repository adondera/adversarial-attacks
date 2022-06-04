"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
Credits: Code was adapted from https://github.com/A-LinCui/Adversarial_Patch_Attack
"""

import matplotlib
import matplotlib.pyplot as plt
import os
from torchvision import models

from attack import patch_attack
from patch_utils import *
from utils import *
from adv_patch_log import AdversarialPatchLog
from config import default_params

matplotlib.use("Qt5agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("training_pictures", exist_ok=True)

logs = AdversarialPatchLog()

# Load the model
model = models.resnet50(pretrained=True).cuda()
model.to(device)
model.eval()

config = default_params()
config["train_size"] = 20
config["test_size"] = 20 # TODO Replace these with higher numbers

# Load the datasets
train_loader, test_loader = dataloader(
    train_size=config["train_size"],
    test_size=config["test_size"],
    data_dir="imagenet_val",
    batch_size=config["batch_size"],
    device=device,
    total_num=50000)

# Test the accuracy of model on trainset and testset
# train_acc, test_acc = test(model, train_loader), test(model, test_loader)
# print('Accuracy of the model on clean train and test sets is {:.3f}% and {:.3f}%'.format(100 * train_acc,
#                                                                                          100 * test_acc))


# Initialize the patch
patch = patch_initialization(config["patch_shape"], image_size=(3, 224, 224), noise_percentage=config["patch_size"])
print('The shape of the patch is', patch.shape)

best_patch_epoch, best_patch_success_rate = 0, 0

# Generate the patch
for epoch in tqdm.tqdm(range(config["epochs"])):
    train_actual_total, train_success = 0, 0
    for (image, label) in tqdm.tqdm(train_loader):
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0].item() != label.item() and predicted[0].data.cpu().numpy() != config["target"]:
            train_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(config["patch_shape"], patch,
                                                                          image_size=(3, 224, 224))
            changed_image, applied_patch = patch_attack(image, applied_patch, mask, config["target"],
                                                        config["probability_threshold"], model, config["lr"],
                                                        config["max_iteration"])
            changed_image = torch.from_numpy(changed_image).to(device)
            output = model(changed_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == config["target"]:
                train_success += 1
            patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
    plt.savefig("training_pictures/" + str(epoch) + " patch.png")
    print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch,
                                                                           100 * train_success / train_actual_total))
    train_success_rate = test_patch(config["patch_shape"], config["target"], patch, test_loader, model)
    print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
    test_success_rate = test_patch(config["patch_shape"], config["target"], patch, test_loader, model)
    print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

    # Record the statistics
    logs.train_scores.append(train_success_rate)
    logs.test_scores.append(test_success_rate)

    # Load the statistics and generate the line
    logs.plot()
    logs.save_log()

    if test_success_rate > best_patch_success_rate:
        best_patch_success_rate = test_success_rate
        best_patch_epoch = epoch
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig("training_pictures/best_patch.png")

print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch,
                                                                                    100 * best_patch_success_rate))
