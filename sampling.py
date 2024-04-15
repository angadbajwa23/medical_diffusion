import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

from ddpm import Diffusion

# n: number of images to sample

device = "cpu"
model = UNet().to(device)
ckpt = torch.load("unconditional_ckpt.pt",map_location=torch.device('cpu'))
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device,noise_steps=1000)
# x = diffusion.sample(model, n=1)
# #plot_images(x)
# save_images(x,'results/individual.png')

# generate 10 images
for i in range(1):
    x = diffusion.sample(model, n=1)
    save_images(x,f'results/large_{i}.png')
