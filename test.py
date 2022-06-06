import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

import torch

model = models.resnet50(pretrained=True)
model.eval()
image = Image.open('train/cat.5.jpg')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]
)

x = torch.rand((3, 70, 70))

tensor = transform(image)
plt.imshow(tensor.permute(1, 2, 0).detach().numpy())
plt.show()
applied_patch = torch.zeros(tensor.shape)
applied_patch[:, 0:70, 0:70] = x
applied_patch.requires_grad = True
optim = torch.optim.SGD(params=[applied_patch], lr=1)
mask = torch.where(applied_patch > 0,
                   torch.ones(applied_patch.shape),
                   torch.zeros(applied_patch.shape))
for i in range(100):
    processed = mask * applied_patch + (1 - mask) * tensor
    input_batch = processed.unsqueeze(0)
    output = model(input_batch)
    probabilities = torch.softmax(output[0], dim=0)
    optim.zero_grad()
    loss = -torch.log(probabilities[254])
    loss.backward()
    optim.step()
    top_probabilities = torch.topk(probabilities, 5)
    if i % 10 == 0:
        plt.imshow(applied_patch.permute(1, 2, 0).detach().numpy())
        plt.show()
    print(probabilities[254])
    # print(top_probabilities)
    #
