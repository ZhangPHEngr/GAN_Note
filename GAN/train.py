# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/25/22 9:33 PM
"""
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from GAN.model import Generator, Discriminator
from GAN.utils import *
from tqdm import tqdm

# ----------------------------全局设置-------------------------------
EPOCH = 200
BATCH_SIZE = 128
IMAGE_SIZE = 28 * 28
INPUT_SIZE = 100
LR = 0.0002
B1 = 0.5
B2 = 0.999
SAMPLES = 200
MINIST_PATH = "/media/zhangph/Elements1/dataset/"
# ----------------------------加载数据集-------------------------------
dataset = datasets.MNIST(MINIST_PATH, train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                             ])
                         )
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------加载模型-------------------------------
# 加载生成器和判别器
gene = Generator(INPUT_SIZE, IMAGE_SIZE)
disc = Discriminator(IMAGE_SIZE)

# ----------------------------加载优化器-------------------------------
opt_G = torch.optim.Adam(gene.parameters(), lr=LR, betas=(B1, B2))
opt_D = torch.optim.Adam(disc.parameters(), lr=LR, betas=(B1, B2))
adver_loss = torch.nn.BCELoss()

# ----------------------------开始训练-------------------------------
# CUDA加速
device = "cuda" if torch.cuda.is_available() else "cpu"
gene.to(device)
disc.to(device)
adver_loss.to(device)

valid = torch.ones(BATCH_SIZE, 1, device=device)  # 或者tensor.cuda() tensor.to("cuda")
fake = torch.zeros(BATCH_SIZE, 1, device=device)

for epoch in range(EPOCH):
    bar = tqdm(data_loader)
    for b, (real_imgs, _) in enumerate(bar):
        bar.set_description("epoch:{} batch:{}".format(epoch, b))
        real_imgs = real_imgs.reshape(BATCH_SIZE, -1)
        if device == "cuda":
            real_imgs = real_imgs.cuda()
        if real_imgs.shape[1] != 784:  # 数据集中有些图像不规则，过滤一下
            continue

        # 训练G
        opt_G.zero_grad()
        x = torch.rand(BATCH_SIZE, INPUT_SIZE, device=device)
        gene_imgs = gene(x)  # 728
        g_loss = adver_loss(disc(gene_imgs), valid)
        g_loss.backward()
        opt_G.step()

        # 训练D
        opt_D.zero_grad()
        real_loss = adver_loss(disc(real_imgs), valid)
        fake_loss = adver_loss(disc(gene_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        if b % SAMPLES == 0:
            print("\nG loss:{}  D loss:{}".format(g_loss, d_loss))
            name = "train_res/img/epoch-{}_batch-{}.jpg".format(epoch, b)
            save_image(gene_imgs, name, vis=True)

    torch.save({
        "gene_state_dict": gene.state_dict(),
        "disc_state_dict": disc.state_dict(),
    }, "train_res/pth/{}.pth".format(epoch))


if __name__ == '__main__':
    pass
