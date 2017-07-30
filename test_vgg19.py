import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
import utils
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning from ImageNet')
parser.add_argument('--test', default='./test_data/tiger.jpeg', help='transfer learning file path')
args = parser.parse_args()

vgg19_npy_path = './vgg19.npy'


# Model
class VGGnet(nn.Module):
    def __init__(self):
        super(VGGnet, self).__init__()
        params_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        # Conv 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1.weight.data = torch.FloatTensor(params_dict['conv1_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv1_1.bias.data = torch.FloatTensor(params_dict['conv1_1'][1])

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2.weight.data = torch.FloatTensor(params_dict['conv1_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv1_2.bias.data = torch.FloatTensor(params_dict['conv1_2'][1])

        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Conv 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1.weight.data = torch.FloatTensor(params_dict['conv2_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv2_1.bias.data = torch.FloatTensor(params_dict['conv2_1'][1])

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2.weight.data = torch.FloatTensor(params_dict['conv2_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv2_2.bias.data = torch.FloatTensor(params_dict['conv2_2'][1])

        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Conv 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1.weight.data = torch.FloatTensor(params_dict['conv3_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_1.bias.data = torch.FloatTensor(params_dict['conv3_1'][1])

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2.weight.data = torch.FloatTensor(params_dict['conv3_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_2.bias.data = torch.FloatTensor(params_dict['conv3_2'][1])

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3.weight.data = torch.FloatTensor(params_dict['conv3_3'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_3.bias.data = torch.FloatTensor(params_dict['conv3_3'][1])

        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4.weight.data = torch.FloatTensor(params_dict['conv3_4'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv3_4.bias.data = torch.FloatTensor(params_dict['conv3_4'][1])

        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Conv 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1.weight.data = torch.FloatTensor(params_dict['conv4_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv4_1.bias.data = torch.FloatTensor(params_dict['conv4_1'][1])

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2.weight.data = torch.FloatTensor(params_dict['conv4_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv4_2.bias.data = torch.FloatTensor(params_dict['conv4_2'][1])

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3.weight.data = torch.FloatTensor(params_dict['conv4_3'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv4_3.bias.data = torch.FloatTensor(params_dict['conv4_3'][1])

        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_4.weight.data = torch.FloatTensor(params_dict['conv4_4'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv4_4.bias.data = torch.FloatTensor(params_dict['conv4_4'][1])

        self.pool4 = nn.MaxPool2d(2, stride=2)

        # Conv 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1.weight.data = torch.FloatTensor(params_dict['conv5_1'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv5_1.bias.data = torch.FloatTensor(params_dict['conv5_1'][1])

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2.weight.data = torch.FloatTensor(params_dict['conv5_2'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv5_2.bias.data = torch.FloatTensor(params_dict['conv5_2'][1])

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3.weight.data = torch.FloatTensor(params_dict['conv5_3'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv5_3.bias.data = torch.FloatTensor(params_dict['conv5_3'][1])

        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_4.weight.data = torch.FloatTensor(params_dict['conv5_4'][0]).permute(3, 2, 0, 1).contiguous()
        self.conv5_4.bias.data = torch.FloatTensor(params_dict['conv5_4'][1])

        self.pool5 = nn.MaxPool2d(2, stride=2)

        # Fully Connected Layers
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc6.weight.data = torch.FloatTensor(params_dict['fc6'][0]).permute(1, 0).contiguous()
        self.fc6.bias.data = torch.FloatTensor(params_dict['fc6'][1])

        self.fc7 = nn.Linear(4096, 4096)
        self.fc7.weight.data = torch.FloatTensor(params_dict['fc7'][0]).permute(1, 0).contiguous()
        self.fc7.bias.data = torch.FloatTensor(params_dict['fc7'][1])

        self.fc8 = nn.Linear(4096, 1000)
        self.fc8.weight.data = torch.FloatTensor(params_dict['fc8'][0]).permute(1, 0).contiguous()
        self.fc8.bias.data = torch.FloatTensor(params_dict['fc8'][1])

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.relu(self.conv4_4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.relu(self.conv5_4(x))
        x = self.pool5(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = VGGnet()
model.cuda()
model.eval()

image = utils.load_image(args.test)
image = image[:, :, ::-1]
VGG_MEAN = np.array([103.939, 116.779, 123.68])
image = (image * 255.0) - VGG_MEAN
image = image.transpose(2, 0, 1)
image = image.astype(np.float32)
input = torch.from_numpy(image)
input = input.cuda()
input_var = torch.autograd.Variable(input, volatile=True)

output = model(input_var.unsqueeze(0))
output = output.data.cpu().numpy()
out = torch.autograd.Variable(torch.from_numpy(output))
utils.print_prob(F.softmax(out).data.numpy()[0], './synset.txt')
