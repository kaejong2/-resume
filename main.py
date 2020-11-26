import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size fo the batches")
parser.add_argument("--lr,--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--b1", type=float, default=0.1, help="adam : decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam : decay of second order momentum of gradient")
# parser.add_argument("--latent_dim", type=int, default=96, help="dimensionality of the latent space")
# parser.add_argument("--n_point", type=int, default=2048, help="number of 2-dim point set")
# parser.add_argument("--chnnels",type=int, default=2, help="number of point set channels")
# parser.add_argument("--sample_interval", type=int, default=10, help="interval between point set visualization")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

