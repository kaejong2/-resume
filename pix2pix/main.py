import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
from Model.cycleGAN import Generator, Discriminator

from utils import ReplayBuffer

from utils import Logger
from utils import weights_init_normal
from utils import save
from utils import set_requires_grad

import os

from dataloader import data_loader

from arguments import Arguments



class cycleGAN():
    def __init__(self, args):
        self.args = args
        
        
    

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        for epoch in range(self.args.num_epochs):
            
            for _iter, data in enumerate(self.dataloader):
               
                print("Train : Epoch %04d/ %04d | Batch %04d / %04d | "
                       "Generator A2B %.4f B2A %.4f | "
                       "Discriminator A %.4f B %.4f | "
                       "Cycle A %.4f B %.4f | "
                       "Identity A %.4f B %.4f | " % 
                       (epoch, self.args.num_epochs, _iter, len(self.dataloader),
                       np.mean(loss_G_A2B_train), np.mean(loss_G_B2A_train),
                       np.mean(loss_D_A_train), np.mean(loss_D_B_train),
                       np.mean(loss_cycle_A_train), np.mean(loss_cycle_B_train),
                       np.mean(loss_identity_A_train), np.mean(loss_identity_B_train)))
            

            save(save_ckpt, self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, self.optimizerG, self.optimizerD, epoch)
        

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = cycleGAN(args)
    
    model.run(save_ckpt=args.ckpt_path, result_path=args.result_path)
 
