# -*- coding: UTF-8 -*-
"""
@Project ：  GAN 
@Author:     Zhang P.H
@Date ：     3/27/22 3:36 PM
"""
from CycleGAN.model import *
from CycleGAN.data_provider import *

DATA_ROOT = "/media/zhangph/Elements1/dataset/GAN/horse2zebra"
test_dataset = GANDataset(DATA_ROOT, split="test")

# 生成网络
netG_A2B = ResnetGenerator(input_nc=3, output_nc=3)
netG_B2A = ResnetGenerator(input_nc=3, output_nc=3)

checkpoint = torch.load("test_res/pth/demo.pth")
netG_A2B.load_state_dict(checkpoint['netG_A2B_state_dict'])
netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])

cnt = 0
for real_A, real_B in test_dataset:

    real_A = real_A.unsqueeze(dim=0)
    real_B = real_B.unsqueeze(dim=0)
    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    real_A = real_A[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    fake_A = fake_A[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    real_B = real_B[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    fake_B = fake_B[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    input_tensor_1 = np.hstack((real_A, fake_B))
    input_tensor_2 = np.hstack((real_B, fake_A))
    input_tensor = np.vstack((input_tensor_1, input_tensor_2))
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imshow("s", input_tensor)
    cv2.waitKey(10000)
    cv2.imwrite("train_res/test_res/{}.jpg".format(cnt), input_tensor)


if __name__ == '__main__':
    pass
