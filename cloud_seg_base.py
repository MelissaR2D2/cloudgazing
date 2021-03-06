"""
File for training networks on the default cloud segmentation task.
"""
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

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda', action='store_true')
args = parser.parse_args()
cuda = args.cuda

if cuda:
    print("using cuda")
    assert torch.cuda.is_available()
else:
    print("using cpu")

# params
model_name = "UNetVOCBase"
image_size = 256
batch_size = 24
epochs = 10
learn_rate = 1e-4
params = {}
for param in ["model_name", "image_size", "batch_size", "epochs", "learn_rate"]:
    params[param] = eval(param)

img_tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# plan: transform target into [size x size] class values. Do it by subclassing VOCSegmentation

target_tfms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((image_size, image_size)),
])

# Data
train_dataset = dataloader.SWIMGSEG(train=True, image_size=image_size)
val_dataset = dataloader.SWIMGSEG(train=False, image_size=image_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Model and Optimizer
model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=2)
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
objective = nn.CrossEntropyLoss()

# metrics
print(len(train_dataset))
print(len(train_loader))
print(len(val_dataset))
print(len(val_loader))
# train
train_losses, train_accs, val_losses, val_accs = train.train(model, train_loader, val_loader, objective,
                                                             optimizer, epochs, batch_size, image_size, cuda)

for param in ["train_losses", "train_accs", "val_losses", "val_accs"]:
    params[param] = eval(param)

train.save_train_results(model, params, "/Users/student/PycharmProjects/CS474FinalProject/results/")