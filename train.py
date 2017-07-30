'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as utils_data
from torch.utils.data.sampler import Sampler

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from include.data import get_data_set
from include.logger import Logger
# from models import *
from utils import progress_bar
from torch.autograd import Variable

import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning from ImageNet')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--file', default='vgg19.npy', help='transfer learning file path')
parser.add_argument('--log', default='./tensorboard/transfer2', help='tensorboard directory')
parser.add_argument('--bn', action='store_true', help='batch normalization')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--notrans', '-n', action='store_true', help='remove transfer learning')
args = parser.parse_args()

# Transfer learning file
trans_path = args.file
# Set the logger
logger = Logger(args.log)
# Set batch norm
BATCH_NORM = args.bn

print('Transfer Learning: %r' % (not args.notrans))
print('Log File: %s' % args.log)
# print(trans_path, args.log, BATCH_NORM)

def to_np(x):
    return x.data.cpu().numpy()


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        size = m.weight.size()
        print(size)
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 80:
        lr = 0.001
    elif epoch <= 122:
        lr = 0.0001
    else:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CifarSampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)

    def __iter__(self):
        # print('\tcalling Sampler:__iter__')
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        return iter(perm[:len(perm) / 3])

    def __len__(self):
        # print('\tcalling Sampler:__len__')
        return self.num_samples / 3


# Model
class VGGnet(nn.Module):
    def __init__(self):
        super(VGGnet, self).__init__()
        params_dict = np.load(trans_path, encoding='latin1').item()
        # Conv 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv1_1.weight.data = torch.FloatTensor(params_dict['conv1_1'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv1_1.bias.data = torch.FloatTensor(params_dict['conv1_1'][1])
        if BATCH_NORM:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv1_2.weight.data = torch.FloatTensor(params_dict['conv1_2'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv1_2.bias.data = torch.FloatTensor(params_dict['conv1_2'][1])
        if BATCH_NORM:
            self.conv1_2_bn = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Conv 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv2_1.weight.data = torch.FloatTensor(params_dict['conv2_1'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv2_1.bias.data = torch.FloatTensor(params_dict['conv2_1'][1])
        if BATCH_NORM:
            self.conv2_1_bn = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv2_2.weight.data = torch.FloatTensor(params_dict['conv2_2'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv2_2.bias.data = torch.FloatTensor(params_dict['conv2_2'][1])
        if BATCH_NORM:
            self.conv2_2_bn = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Conv 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv3_1.weight.data = torch.FloatTensor(params_dict['conv3_1'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv3_1.bias.data = torch.FloatTensor(params_dict['conv3_1'][1])
        if BATCH_NORM:
            self.conv3_1_bn = nn.BatchNorm2d(256)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv3_2.weight.data = torch.FloatTensor(params_dict['conv3_2'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv3_2.bias.data = torch.FloatTensor(params_dict['conv3_2'][1])
        if BATCH_NORM:
            self.conv3_2_bn = nn.BatchNorm2d(256)

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv3_3.weight.data = torch.FloatTensor(params_dict['conv3_3'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv3_3.bias.data = torch.FloatTensor(params_dict['conv3_3'][1])
        if BATCH_NORM:
            self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv3_4.weight.data = torch.FloatTensor(params_dict['conv3_4'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv3_4.bias.data = torch.FloatTensor(params_dict['conv3_4'][1])
        if BATCH_NORM:
            self.conv3_4_bn = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Conv 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv4_1.weight.data = torch.FloatTensor(params_dict['conv4_1'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv4_1.bias.data = torch.FloatTensor(params_dict['conv4_1'][1])
        if BATCH_NORM:
            self.conv4_1_bn = nn.BatchNorm2d(512)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv4_2.weight.data = torch.FloatTensor(params_dict['conv4_2'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv4_2.bias.data = torch.FloatTensor(params_dict['conv4_2'][1])
        if BATCH_NORM:
            self.conv4_2_bn = nn.BatchNorm2d(512)

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv4_3.weight.data = torch.FloatTensor(params_dict['conv4_3'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv4_3.bias.data = torch.FloatTensor(params_dict['conv4_3'][1])
        if BATCH_NORM:
            self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv4_4.weight.data = torch.FloatTensor(params_dict['conv4_4'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv4_4.bias.data = torch.FloatTensor(params_dict['conv4_4'][1])
        if BATCH_NORM:
            self.conv4_4_bn = nn.BatchNorm2d(512)

        self.pool4 = nn.MaxPool2d(2, stride=2)

        # Conv 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv5_1.weight.data = torch.FloatTensor(params_dict['conv5_1'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv5_1.bias.data = torch.FloatTensor(params_dict['conv5_1'][1])
        if BATCH_NORM:
            self.conv5_1_bn = nn.BatchNorm2d(512)

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv5_2.weight.data = torch.FloatTensor(params_dict['conv5_2'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv5_2.bias.data = torch.FloatTensor(params_dict['conv5_2'][1])
        if BATCH_NORM:
            self.conv5_2_bn = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv5_3.weight.data = torch.FloatTensor(params_dict['conv5_3'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv5_3.bias.data = torch.FloatTensor(params_dict['conv5_3'][1])
        if BATCH_NORM:
            self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if(not args.notrans):
            self.conv5_4.weight.data = torch.FloatTensor(params_dict['conv5_4'][0]).permute(3, 2, 0, 1).contiguous()
            self.conv5_4.bias.data = torch.FloatTensor(params_dict['conv5_4'][1])
        if BATCH_NORM:
            self.conv5_4_bn = nn.BatchNorm2d(512)

        # self.pool5 = nn.MaxPool2d(2, stride=2)

        # Fully Connected Layers
        self.fc6 = nn.Linear(512 * 2 * 2, 4096)
        if BATCH_NORM:
            self.fc6_bn = nn.BatchNorm1d(4096)

        # self.fc6.weight.data = torch.FloatTensor(params_dict['fc6'][0]).permute(1, 0).contiguous()
        # self.fc6.bias.data = torch.FloatTensor(params_dict['fc6'][1])

        self.fc7 = nn.Linear(4096, 4096)
        if(not args.notrans):
            self.fc7.weight.data = torch.FloatTensor(params_dict['fc7'][0]).permute(1, 0).contiguous()
            self.fc7.bias.data = torch.FloatTensor(params_dict['fc7'][1])
        if BATCH_NORM:
            self.fc7_bn = nn.BatchNorm1d(4096)

        self.fc8 = nn.Linear(4096, 10)
        if BATCH_NORM:
            self.fc8_bn = nn.BatchNorm1d(10)

        # self.fc8.weight.data = torch.FltTensor(params_dict['fc8'][0]).permute(1, 0).contiguous()
        # self.fc8.bias.data = torch.FloatTensor(params_dict['fc8'][1])

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # First hidden layer
        x = self.conv1_1(x)
        if BATCH_NORM:
            x = self.conv1_1_bn(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        if BATCH_NORM:
            x = self.conv1_2_bn(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        if BATCH_NORM:
            x = self.conv2_1_bn(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        if BATCH_NORM:
            x = self.conv2_2_bn(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        if BATCH_NORM:
            x = self.conv3_1_bn(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        if BATCH_NORM:
            x = self.conv3_2_bn(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        if BATCH_NORM:
            x = self.conv3_3_bn(x)
        x = F.relu(x)
        x = self.conv3_4(x)
        if BATCH_NORM:
            x = self.conv3_4_bn(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        if BATCH_NORM:
            x = self.conv4_1_bn(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        if BATCH_NORM:
            x = self.conv4_2_bn(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        if BATCH_NORM:
            x = self.conv4_3_bn(x)
        x = F.relu(x)
        x = self.conv4_4(x)
        if BATCH_NORM:
            x = self.conv4_4_bn(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        if BATCH_NORM:
            x = self.conv5_1_bn(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        if BATCH_NORM:
            x = self.conv5_2_bn(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        if BATCH_NORM:
            x = self.conv5_3_bn(x)
        x = F.relu(x)
        x = self.conv5_4(x)
        if BATCH_NORM:
            x = self.conv5_4_bn(x)
        x = F.relu(x)
        # x = self.pool5(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc6(x)
        if BATCH_NORM:
            x = self.fc6_bn(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc7(x)
        if BATCH_NORM:
            x = self.fc7_bn(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc8(x)
        if BATCH_NORM:
            x = self.fc8_bn(x)
        x = F.relu(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_x, train_y, train_l = get_data_set(cifar=10)
test_x, test_y, test_l = get_data_set(name="test", cifar=10)

train_x = train_x.astype(np.float32, copy=False)
test_x = test_x.astype(np.float32, copy=False)

x = torch.from_numpy(train_x)
y = torch.from_numpy(train_y)
trainset = utils_data.TensorDataset(x, y)
sampler = CifarSampler(trainset)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=sampler, shuffle=False, num_workers=2)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

x = torch.from_numpy(test_x)
y = torch.from_numpy(test_y)
testset = utils_data.TensorDataset(x, y)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

step = 0

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = VGGnet()
    # net.apply(weight_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # print('B')
        # print (outputs[0])
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global step
    global best_acc
    step = step + 1
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    # TensorBoard Logging
    # (1) Log Scalle
    info = {
        'loss': test_loss,
        'accuracy': 100. * correct / total
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step + 1)
    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), step + 1)
        logger.histo_summary(tag + '/grad', to_np(value.grad), step + 1)

    # (3) Log the images
    info = {
        'images': to_np(inputs.view(-1, 3, 32, 32)[:10])
    }
    for tag, images in info.items():
        logger.image_summary(tag, images, step + 1)


for epoch in range(start_epoch, start_epoch + 164):
    train(epoch)
    test(epoch)
