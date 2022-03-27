# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/25/22 11:43 PM
"""
import os.path

import cv2
import numpy as np


def save_image(data, name, vis=False):
    data = data[:25]
    gene_imgs = data.reshape(-1, 28, 28)
    data = gene_imgs.detach().cpu().numpy()

    img = np.zeros((5 * 28, 5 * 28))
    for i in range(5):
        for j in range(5):
            img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = data[i * 5 + j, :, :]

    img = cv2.resize(img, (640, 640))
    cv2.imwrite(name, img)

    if vis:
        cv2.imshow("s", img)
        cv2.waitKey(100)


if __name__ == '__main__':
    pass
