import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import train
import dataloader
import argparse

"""
File for testing pretrained object segmentation networks on images of clouds.
Goal is to see what confidence pretrained networks assign to the "objects" and
if the segmentation maps look realistic
"""

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
    "outline"
]

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--grayscale', dest='grayscale', action='store_true')
args = parser.parse_args()
cuda = args.cuda
grayscale = args.grayscale

if cuda:
    print("using cuda")
    assert torch.cuda.is_available()
else:
    print("using cpu")

print("Grayscale images: ", grayscale)

# params
model_name = "UNetVOCBaseGrayscale"
image_size = 256
batch_size = 24
epochs = 10
learn_rate = 1e-4
params = {}
for param in ["model_name", "image_size", "batch_size", "epochs", "learn_rate"]:
    params[param] = eval(param)

print(model_name)

# Data
train_dataset = dataloader.VOC("/Users/student/Documents/College/", image_set='train', image_size=image_size, grayscale=grayscale)
val_dataset = dataloader.VOC("/Users/student/Documents/College/", image_set='val', image_size=image_size, grayscale=grayscale)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Model and Optimizer
model = smp.Unet(encoder_weights="imagenet", classes=len(VOC_CLASSES))
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
objective = nn.CrossEntropyLoss()

# metrics
print(len(val_dataset))
print(len(val_loader))
# train
train_losses, train_accs, val_losses, val_accs = train.train(model, train_loader, val_loader, objective,
                                                             optimizer, epochs, batch_size, image_size, cuda)

for param in ["train_losses", "train_accs", "val_losses", "val_accs"]:
    params[param] = eval(param)

train.save_train_results(model, params, "./results/")