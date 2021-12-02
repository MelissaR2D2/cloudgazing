import torch
from torchvision import transforms
import numpy as np
from dataloader import VOC_CLASSES, VOC_COLORMAP

def create_cloud_mask(image, seg):
    image = np.ndarray.copy(image.numpy())
    locations = np.where(seg == 0)
    image[:, locations[0], locations[1]] = 0
    return torch.from_numpy(image)

def create_segmented_image(image, result):
    image = np.asarray(transforms.ToPILImage()(image))
    for key, value in VOC_COLORMAP.items():
        if value == 0 or value == 21:
            # background and outline colors
            continue
        locations = np.where(result == value)
        image[locations[0], locations[1]] = np.asarray(key)
        """ image[0, locations[0], locations[1]] = key[0]
        image[1, locations[0], locations[1]] = key[1]
        image[2, locations[0], locations[1]] = key[2]"""
    return image
