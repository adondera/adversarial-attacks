import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
out_pixeled = torch.load("save\images_resnet50_3.pth")
transform = T.ToPILImage()
first = out_pixeled['images'][2]
# for i in range(20):
#     tens = out_pixeled['images'][i]
#     print(tens)
#     img = transform(tens)
#     img.show()
#     img.save("non_attacked_images/na"+str(i)+".jpg")
# softmax = torch.nn.Softmax()
# resnet18 = models.resnet18(pretrained=True)
# input_var = torch.autograd.Variable(first, volatile=True)
# print(resnet18.forward(input_var))
# output = softmax.forward(resnet18.forward(input_var))
transform = T.ToPILImage()
img = transform(first)
# img.save("non_attacked_images/na1.jpg")
img.show()