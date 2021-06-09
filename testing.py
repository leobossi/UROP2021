# Testing Pytorch works and completeing the exercies outlined in the Deep Learning with Pytorch textbook.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:52:44 2021

@author: leonardobossi1
"""

import torch
import numpy as np

from torchvision import models
from torchvision import transforms

# running AlexNet architecture on input image 
alexnet = models.AlexNet()

# download the weights of resnet101 trained on the ImageNet dataset
# when printing resnet, we see the individual modules (layers) that build the neural network

resnet = models.resnet101(pretrained=True)

# Preprocessing so that the input images have the right parameters
preprocess = transforms.Compose([
transforms.Resize(256),           # scaling
transforms.CenterCrop(224),       # cropping
transforms.ToTensor(),            # transform to a tensor
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])

from PIL import Image

# example image of a dog

img2 = Image.open("dog1.jpeg")
img = Image.open('dlwpt-code-master/data/p1ch2/bobby.jpg')

img_t2 = preprocess(img2)
img_t = preprocess(img)

# Normalising the tensor
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval() # putting network to eval mode

out = resnet(batch_t) # tensor of 1000 scores, one per image class

with open('dlwpt-code-master/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Finding the max index in the output tensor
index = torch.max(out, 1)

# Using torch.nn.functional.softmax to output confidence level based on the 1000 scores

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[1]], percentage[index[1]].item()

# Checking the top choices for the classification
indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[1][0][:5]]


#%%
'''
# Working with CycleGAN (converting images of horses to zebras)

netG = ResNetGenerator()

model_path = 'dlwpt-code-master/data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

netG.eval()

'''

#%%
# Introduction to tensors and indexing

list_a = [1, 2, 1]

a = torch.ones(3)
print(float(a[1]))

# Changing one of the elements
a[2] = 2
print(a)

# Storing information about a triangle with (4, 1) (5,3) (2,1)
points = torch.zeros(6)

# Even indices represent x coordinates of vertices
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

#Alernatively:
points2 = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])

# Switching to more intuitive, 2D tensors:
points2D = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points2D.shape

points2D[0,1]
points2D[0]

# Example with turning an image into grayscale
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# Increasing to a batch of images
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]

# Averages the RGB channel 
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)


# Broadcasting (considering the weights too)
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)  # shape is changed from [3] to [3,1,1]

# Weighting and then taking the sum 
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)

img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)


# Adding names
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

weights_aligned = weights_named.align_as(img_named)
weights_aligned.names # Names of the weights are aligned according to img_named

# Can carry out operations using names
gray_named = (img_named * weights_aligned).sum('channels')
gray_named.names


# 32- bit floating-point is standard dtype for tensors
# if tensor with integers is created, int64 bit is the standard dtype

# To switch between dtypes
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

short_points.dtype

# Storage 
points3 = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points4 = torch.tensor([[4.0, 1.0, 5.0],[3.0 ,2.0 ,1.0]])

points3.storage()
points4.storage()

t_points4 = points4.t() # Used to transpose 2D tensors

#%%
points = torch.tensor([[4.0, 1.0], [5.0, 2.0], [2.0, 1.0]])
second_point = points[2]

second_point.storage()
second_point.storage_offset()


# storing onto GPU
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')

# alernatively, points_gpu = points.to(device='cuda')

# To convert from PyTorch tensor to numpy and vice versa, points = torch.from_numpy(points_np)

a = torch.tensor(list(range(9)))

b = a.view(3, 3)

c = b[1:,1:]

#%%
# Reading images
import imageio

img_arr = imageio.imread('dlwpt-code-master/data/p1ch4/image-dog/bobby.jpg')
img_arr.shape

img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1) # channel 2 first, then 0 and 1

# for multiple images, N × C × H × W (number, channel, height, width)


#%%
import csv

wine_path = "dlwpt-code-master/data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
skiprows=1)

col_list = next(csv.reader(open(wine_path), delimiter=';'))

#Converting to torch tensor
wineq = torch.from_numpy(wineq_numpy)









