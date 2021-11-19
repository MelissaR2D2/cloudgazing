import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import train

"""
File for testing pretrained object segmentation networks on images of clouds.
Goal is to see what confidence pretrained networks assign to the "objects" and
if the segmentation maps look realistic
"""

def decide(output):
    return np.argmax(output.detach().numpy(), axis=1)

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
]

# params
model_name = "UNetVOCBase"
image_size = 300
batch_size = 20
epochs = 10
learn_rate = 1e-4
params = {}
for param in ["model_name", "image_size", "batch_size", "epochs", "learn_rate"]:
    params[param] = eval(param)

tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Data
train_dataset = datasets.VOCSegmentation("/Users/student/Documents/College/", download=True, image_set='train', transforms=tfms)
val_dataset = datasets.VOCSegmentation("/Users/student/Documents/College/", download=True, image_set='val', transforms=tfms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Model and Optimizer
model = smp.Unet(encoder_weights="imagenet")
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
objective = nn.CrossEntropyLoss()

# train
train_losses, train_accs, val_losses, val_accs = train.train(model, train_loader, val_loader, objective,
                                                             optimizer, epochs, batch_size, image_size)

for param in ["train_losses", "train_accs", "val_losses", "val_accs"]:
    params[param] = eval(param)

train.save_train_results(model, params, "/Users/student/PycharmProjects/CS474FinalProject/results/")