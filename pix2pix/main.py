import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
import os

from dataloader import data_loader

from arguments import Arguments

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Model.pix2pix import Generator, Discriminator

from utils import init_weight
from utils import pix2pix_save
from utils import set_requires_grad





class pix2pix():
    def __init__(self, args):
        self.args = args
        
        self.G = Generator(input_channels=3).to(device= self.args.device)
        self.D = Discriminator(output_channels=1).to(device= self.args.device)
        init_weight(self.G, init_type="normal", init_gain=0.02)
        init_weight(self.D, init_type="normal", init_gain=0.02)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
        self.criterion_GAN = torch.nn.BCELoss().to(device=args.device)
        self.criterion_L1 =  torch.nn.L1Loss().to(device=args.device)

        dataset_train = data_loader(args.data_path+"train", nch=3)
        dataset_val = data_loader(args.data_path+"val", nch=3)

        self.data_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        self.data_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)


    def train(self, ckpt_path=None, result_path=None):
        for epoch in range(self.args.num_epochs):
            self.G.train()
            self.D.train()

            Loss_G_GAN = []
            Loss_G_l1 = []
            Loss_D_real = []
            Loss_D_fake = []

            for _iter, data in enumerate(self.data_train):
                data = data["input"].to(device= self.args.device)
                label = data["output"].to(device= self.args.device)

                output = self.G(data)
                set_requires_grad(self.D, True)
                self.optimizerD.zero_grad

                real = torch.cat([data, label], dim=1)
                fake = torch.cat([data, output], dim=1)

                pred_real = self.D(real)
                pred_fake = self.D(fake.detach())

                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5*loss_D_fake + 0.5*loss_D_real
                loss_D.backward()
                self.optimizerD.step()
                set_requires_grad(self.D, False)

                self.optimizerG.zero_grad
                fake = torch.cat([data, output], dim=1)
                pred_fake = self.D(fake)

                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_real))
                loss_G_l1 = self.criterion_L1(output, label)
                loss_G = loss_G_GAN + 100*loss_G_l1

                loss_G.backward()
                self.optimizerG.step()

                Loss_G_GAN += [loss_G_GAN.item()]
                Loss_G_l1 += [loss_G_GAN.item()]
                Loss_D_fake += [loss_D_fake.item()]
                Loss_D_real += [loss_D_real.item()]

                
               
                print("Train : Epoch %04d/ %04d | Batch %04d / %04d | "
                       "Generator G GAN %.4f | "
                       "Generator G L1 %.4f | "
                       "Discriminator Real %.4f | "
                       "Discriminator Fake %.4f | "
                       (epoch, self.args.num_epochs, _iter, len(self.data_train),
                       np.mean(Loss_G_GAN), np.mean(Loss_G_l1),
                       np.mean(Loss_D_real), np.mean(Loss_D_fake)))
        
            pix2pix_save(ckpt_path, self.G, self.D, self.optimizerG, self.optimizerD, epoch)
        

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = pix2pix(args)
    
    model.train(ckpt_path=args.ckpt_path, result_path=args.result_path)
 
