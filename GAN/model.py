# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/25/22 9:57 PM
"""
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim, image_shape):
        super().__init__()
        self._input_dim = input_dim
        self._image_shape = image_shape
        self.module = nn.Sequential(
            nn.Linear(self._input_dim, 128), nn.BatchNorm1d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256), nn.BatchNorm1d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self._image_shape), nn.Tanh()
        )

    def forward(self, x):
        return self.module(x)


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.model(image)


def save_image(data):
    pass

if __name__ == '__main__':
    pass
