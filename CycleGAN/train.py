# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/26/22 11:37 PM
"""
import os.path

import cv2
import numpy as np
import torch
from torch import optim
import itertools
from torch.utils.data import DataLoader
from CycleGAN.model import ResnetGenerator, NLayerDiscriminator
from CycleGAN.data_provider import GANDataset

# 全局设置
DATA_ROOT = "/media/zhangph/Elements1/dataset/GAN/horse2zebra"
EPOCH = 200
BATCH_SIZE = 2
SAMPLES = 200
LR = 0.0002
BETA1 = 0.5
lambdaIdt = 0.5
lambdaA = 10.0
lambdaB = 10.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据
train_dataset = GANDataset(DATA_ROOT, split="train")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 生成网络
netG_A2B = ResnetGenerator(input_nc=3, output_nc=3).to(device)
netG_B2A = ResnetGenerator(input_nc=3, output_nc=3).to(device)

# 判决网络
netD_A = NLayerDiscriminator(input_nc=3).to(device)
netD_B = NLayerDiscriminator(input_nc=3).to(device)
# print(netD_B)

# loss
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_ionIdt = torch.nn.L1Loss().to(device)

# 优化器
optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=LR, betas=(BETA1, 0.999))

# Target Setting
true_target = torch.ones((BATCH_SIZE, 1, 30, 30)).to(device)
false_target = torch.zeros((BATCH_SIZE, 1, 30, 30)).to(device)


def get_G_loss(real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B):
    # loss 1
    loss_g1_A = criterion_GAN(netD_A(fake_A), true_target)
    loss_g1_B = criterion_GAN(netD_B(fake_B), true_target)
    # loss 2 : cycle loss
    loss_g2_A = criterion_cycle(real_A, rec_A) * lambdaA
    loss_g2_B = criterion_cycle(real_B, rec_B) * lambdaB
    # loss 3
    loss_g3_A = criterion_ionIdt(real_A, idt_A) * lambdaA * lambdaIdt
    loss_g3_B = criterion_ionIdt(real_B, idt_B) * lambdaB * lambdaIdt

    loss_g = loss_g1_A+loss_g1_B+loss_g2_A+loss_g2_B+loss_g3_A+loss_g3_B
    return loss_g

def get_D_loss(real_A, fake_A, real_B, fake_B):
    # A
    loss_d_A_real = criterion_GAN(netD_A(real_A), true_target)
    loss_d_A_fake = criterion_GAN(netD_A(fake_A.detach()), false_target)
    loss_d_A = (loss_d_A_real + loss_d_A_fake) * 0.5
    # B
    loss_d_B_real = criterion_GAN(netD_B(real_B), true_target)
    loss_d_B_fake = criterion_GAN(netD_B(fake_B.detach()), false_target)
    loss_d_B = (loss_d_B_real + loss_d_B_fake) * 0.5

    return loss_d_A, loss_d_B


for epoch in range(EPOCH):
    for idx, (real_A, real_B) in enumerate(train_dataloader):
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        fake_A = netG_B2A(real_B)
        fake_B = netG_A2B(real_A)
        rec_A = netG_B2A(fake_B)
        rec_B = netG_A2B(fake_A)
        idt_A = netG_B2A(real_A)
        idt_B = netG_A2B(real_B)

        # G_A and G_B
        netD_A.requires_grad_(False)
        netD_B.requires_grad_(False)
        optimizer_G.zero_grad()
        loss_g = get_G_loss(real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B)
        loss_g.backward()
        optimizer_G.step()

        # D_A and D_B
        netD_A.requires_grad_(True)
        netD_B.requires_grad_(True)
        optimizer_D.zero_grad()
        loss_d_A, loss_d_B = get_D_loss(real_A, fake_A, real_B, fake_B)
        loss_d_A.backward()
        loss_d_B.backward()
        optimizer_D.step()

        if idx % SAMPLES == 0:
            print("epoch:{} batch:{} g_loss:{} d_loss:{}".format(epoch, idx, loss_g, loss_d_A+loss_d_B))
            real_A = real_A[0].cpu().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            fake_A = fake_A[0].cpu().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            real_B = real_B[0].cpu().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            fake_B = fake_B[0].cpu().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            input_tensor_1 = np.hstack((real_A, fake_B))
            input_tensor_2 = np.hstack((real_B, fake_A))
            input_tensor = np.vstack((input_tensor_1, input_tensor_2))
            # RGB转BRG
            input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
            cv2.imshow("s", input_tensor)
            cv2.waitKey(200)
            cv2.imwrite("train_res/image/epoch-{}_batch-{}.jpg".format(epoch, idx), input_tensor)
    torch.save({
        "netG_A2B_state_dict": netG_A2B.state_dict(),
        "netG_B2A_state_dict": netG_B2A.state_dict(),
        "netD_A_state_dict": netD_A.state_dict(),
        "netD_B_state_dict": netD_B.state_dict(),
    }, "train_res/pth/{}.pth".format(epoch))

if __name__ == '__main__':
    pass
