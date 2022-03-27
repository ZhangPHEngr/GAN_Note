# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/26/22 11:51 PM
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((286, 286)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


class GANDataset(Dataset):
    def __init__(self, root, split="train"):
        image_A_dir = os.path.join(root, split + "A")
        image_B_dir = os.path.join(root, split + "B")
        assert os.path.isdir(image_A_dir), '%s is not a valid directory' % image_A_dir

        self.images_A_path = list()
        for root, _, fnames in sorted(os.walk(image_A_dir)):
            self.images_A_path = [os.path.join(root, fname) for fname in fnames]

        self.images_B_path = list()
        for root, _, fnames in sorted(os.walk(image_B_dir)):
            self.images_B_path = [os.path.join(root, fname) for fname in fnames]

    def __getitem__(self, index):
        img_A = Image.open(self.images_A_path[index])
        img_B = Image.open(self.images_B_path[index])
        return preprocess(img_A.convert("RGB")), preprocess(img_B.convert("RGB"))

    def __len__(self):
        return min(len(self.images_A_path), len(self.images_B_path))


if __name__ == '__main__':
    data = GANDataset("/media/zhangph/Elements1/dataset/GAN/horse2zebra", split="train")
    print(data[1][0])
    for i in range(data.__len__()):
        data_A = data[i][0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        data_B = data[i][1].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        # RGB转BRG
        input_tensor = np.hstack((data_A,data_B))
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
        cv2.imshow("s", input_tensor)
        cv2.waitKey(200)
