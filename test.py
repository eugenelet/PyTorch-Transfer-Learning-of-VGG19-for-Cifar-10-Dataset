import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning from ImageNet')
parser.add_argument('--test', default='./test_data/tiger.jpeg', help='transfer learning file path')
args = parser.parse_args()


vgg19_npy_path = './vgg19.npy'

model = models.vgg19(pretrained=True)
# model = models.alexnet(pretrained=True)
# model = VGGnet()
model.cuda()
model.eval()

# image = cv2.imread('tiny-imagenet-200/test/images/test_10.JPEG')
# image = cv2.imread('test_data/tiger.jpeg')
image = utils.load_image(args.test)
# image.resize(224, 224, 3)
# image = image[:, :, ::-1]
# image = image / 255.0
VGG_MEAN = np.array([0.485, 0.456, 0.406])
VGG_STD = np.array([0.229, 0.224, 0.225])
# image = (image * 255.0) - VGG_MEAN
image = (image - VGG_MEAN) / VGG_STD
# image = image.reshape(3, 224, 224)
image = image.transpose(2, 0, 1)
# image = image.reshape(3, 64, 64)
image = image.astype(np.float32)
input = torch.from_numpy(image)
input = input.cuda()
input_var = torch.autograd.Variable(input, volatile=True)

output = model(input_var.unsqueeze(0))
output = output.data.cpu().numpy()
output = torch.autograd.Variable(torch.from_numpy(output))
utils.print_prob(F.softmax(output).data.numpy()[0], './synset.txt')
# prob = np.argsort(output)[::-1]

# utils.print_prob(prob[0], './synset.txt')
