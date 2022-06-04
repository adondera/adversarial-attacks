"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
Credits: Code was adapted from https://github.com/A-LinCui/Adversarial_Patch_Attack
"""
from torch.autograd import Variable

from utils import *


# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbed picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    changed_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
        (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        changed_image = Variable(changed_image.data, requires_grad=True)
        per_image = changed_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = changed_image.grad.clone().cpu()
        changed_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        changed_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
            (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        changed_image = torch.clamp(changed_image, min=-3, max=3)
        changed_image = changed_image.cuda()
        output = model(changed_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    changed_image = changed_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return changed_image, applied_patch
