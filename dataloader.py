import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random


class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, 'mask', os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = None  # use opencv or pil read image using img_path
        label = None  # use opencv or pil read label  using mask_path
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


"""
Splits data into train and test sets.
Images are set up like this:

"""


def print_dataset_statistics():
    img_names_len = len(glob.glob("/Users/student/Documents/College/swimseg/images/d*"))
    mask_names_len = len(glob.glob("/Users/student/Documents/College/swimseg/masks/d*"))
    print("{} total images, {} unique".format(img_names_len, img_names_len / 6.0))
    print("{} total masks, {} unique".format(mask_names_len, mask_names_len / 6.0))


"""
Splits dataset with a train/test split of 80/20.
Result: 
train:
- 810 unique images (4860 total)
test:
- 203 unique images (1218 total)
File names: d0000
"""


def split_dataset():
    unique = 1013
    test_total = 203
    nums = [format(num, '04') for num in random.sample(range(unique), test_total)]
    print(len(nums))
    imgs_src_prefix = "/Users/student/Documents/College/swimseg/images/"
    masks_src_prefix = "/Users/student/Documents/College/swimseg/masks/"
    train_imgs_dest_prefix = "/Users/student/Documents/College/swimseg_split/train/images/"
    train_masks_dest_prefix = "/Users/student/Documents/College/swimseg_split/train/masks/"
    test_imgs_dest_prefix = "/Users/student/Documents/College/swimseg_split/test/images/"
    test_masks_dest_prefix = "/Users/student/Documents/College/swimseg_split/test/masks/"

    imgs_moved = 0
    masks_moved = 0
    for num in nums:
        imgs_found = glob.glob(imgs_src_prefix + "d" + num + "*")
        masks_found = glob.glob(masks_src_prefix + "d" + num + "*")
        if len(imgs_found) == 6 and len(masks_found) == 6:
            for file in imgs_found:
                os.renames(file, test_imgs_dest_prefix + os.path.basename(file))
                imgs_moved += 1
            for file in masks_found:
                os.renames(file, test_masks_dest_prefix + os.path.basename(file))
                masks_moved += 1
        else:
            print("Error: missing imgs or masks at num {}".format(num))
    print("{} test images moved".format(imgs_moved))
    print("{} test masks moved".format(masks_moved))

    # gets the rest of the imgs and masks, which will be train data
    rest_imgs = glob.glob(imgs_src_prefix + "d*")
    rest_masks = glob.glob(masks_src_prefix + "d*")
    for img, mask in zip(rest_imgs, rest_masks):
        os.renames(img, train_imgs_dest_prefix + os.path.basename(img))
        os.renames(mask, train_masks_dest_prefix + os.path.basename(mask))
    print("{} train images moved".format(len(rest_imgs)))
    print("{} train masks moved".format(len(rest_masks)))

    total_train_imgs = len(glob.glob(train_imgs_dest_prefix + "*"))
    total_train_masks = len(glob.glob(train_masks_dest_prefix + "*"))
    total_test_imgs = len(glob.glob(test_imgs_dest_prefix + "*"))
    total_test_masks = len(glob.glob(test_masks_dest_prefix + "*"))
    print("Final results:")
    print("{} train imgs".format(total_train_imgs))
    print("{} train masks".format(total_train_masks))
    print("{} test imgs".format(total_test_imgs))
    print("{} test masks".format(total_test_masks))


if __name__ == "__main__":
    # split_dataset()
    pass
