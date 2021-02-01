import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
import os
from torchvision.utils import save_image
from dataloader import data_loader
from PIL import Image

from arguments import Arguments

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Model.pix2pix import Generator, Discriminator

from utils import init_weight
from utils import pix2pix_save, pix2pix_load
from utils import set_requires_grad



# def save_image(image_tensor):
#     img = image_tensor.to('cpu').detach().numpy().transpose(0,2,3,1)
#     img = img/2.0 *255.0
#     img = img.clip(0,255)
#     img = img.astype(np.uint8)
    
#     return img

class pix2pix():
    def __init__(self, args):
        self.args = args
        
        self.G = Generator(input_channels=3).to(device=self.args.device)
        self.D = Discriminator(output_channels=1).to(device=self.args.device)
        init_weight(self.G, init_type="normal", init_gain=0.02)
        init_weight(self.D, init_type="normal", init_gain=0.02)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device=args.device)
        self.criterion_L1 =  torch.nn.L1Loss().to(device=args.device)
        self.transform_train = transforms.Compose([transforms.Resize((286,286), Image.BICUBIC), transforms.RandomCrop(256), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.transform_test = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = data_loader(args.data_path+"train", transform=self.transform_train)
        dataset_val = data_loader(args.data_path+"val", transform=self.transform_test)

        self.data_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        self.data_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

        # self.ToNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)

    def train(self, ckpt_path=None, result_path=None):
        for epoch in range(self.args.epoch, self.args.num_epochs):
            self.G.train()
            self.D.train()
            for _iter, (A,B) in enumerate(self.data_train):
                data = Variable(A.float()).to(device=self.args.device)
                label = Variable(B.float()).to(device=self.args.device)
                output = self.G(data)
                                
                #################################################
                #              Train Discriminator
                #################################################
                self.optimizerD.zero_grad

                real = torch.cat([data, label], dim=1)
                fake = torch.cat([data, output.detach()], dim=1)

                pred_real = self.D(real)
                pred_fake = self.D(fake)

                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_fake + loss_D_real)

                loss_D.backward()
                self.optimizerD.step()

                #################################################
                #              Train Generator
                #################################################
                self.optimizerG.zero_grad
                fake = torch.cat([data, output], dim=1)
                pred_fake = self.D(fake)

                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_real))
                loss_G_l1 = self.criterion_L1(output, label)
                loss_G = loss_G_GAN + 100*loss_G_l1

                loss_G.backward()
                self.optimizerG.step()

                #################################################
                #              Training Process
                #################################################               
                print("Train : Epoch %02d/ %02d | Batch %03d / %03d | "
                       "Generator G GAN %.4f | "
                       "Generator G L1 %.4f | "
                       "Discriminator Real %.4f | "
                       "Discriminator Fake %.4f | " %
                       (epoch, self.args.num_epochs, _iter, len(self.data_train),
                       loss_G_GAN, loss_G_l1,
                       loss_D_real, loss_D_fake))
        
            pix2pix_save(ckpt_path, self.G, self.D, self.optimizerG, self.optimizerD, epoch)
            if epoch % 10==0:
                print("Sample Save")
                self.sample_save(result_path=self.args.sample_path)
                self.sample_image(result_path=self.args.result_path)
    def sample_save(self, result_path):
        with torch.no_grad():
            self.G.eval()
            for _iter, (A, B) in enumerate(self.data_val):
                data = Variable(A.float()).to(device=self.args.device)
                label = Variable(B.float()).to(device=self.args.device)
                output = self.G(data)
                for i in range(data.shape[0]):
                    name = args.batch_size * (_iter) + i
                    fileset = {'name': name,
                               'data': "%04d-data.png"%name,
                               'output': "%04d-output.png"%name,
                               'label': "%04d-label.png"%name}
                    save_image(data[i], os.path.join(result_path, fileset['data']), normalize=True)
                    save_image(label[i], os.path.join(result_path, fileset['label']), normalize=True)
                    save_image(output[i], os.path.join(result_path, fileset['output']), normalize=True)
    def sample_image(self, result_path):
        A, B = next(iter(self.data_val))
        data1 = Variable(A.float()).to(device=self.args.device)
        label = Variable(B.float()).to(device=self.args.device)
        output = self.G(data1)
        img_sample = torch.cat((data1.data, output.data, label.data), -2)
        save_image(img_sample, os.path.join(result_path, "data.png"), normalize=True)


                


    # def test(self, ckpt_path=None, result_path=None):
    #     pix2pix_load(ckpt_path, self.G, self.D, self.optimizerG, self.optimizerD, epoch=1)
    #     with torch.no_grad():
    #         self.G.eval()

    #         for _iter, (A, B) in enumerate(self.data_val):
    #             data = Variable(A.float()).to(device=self.args.device)
    #             label = Variable(B.float()).to(device=self.args.device)

    #             output = self.G(data)

    #             # loss_G_l1 = self.criterion_L1(output, label)
    #             # loss_G = loss_G_GAN + 100*loss_G_l1

    #             data = save_image(data)
    #             label = save_image(label)
    #             output = save_image(output)

    #             for i in range(data.shape[0]):
    #                 name = args.batch_size * (_iter) + i
    #                 fileset = {'name': name,
    #                            'data': "%04d-data.png"%name,
    #                            'output': "%04d-output.png"%name,
    #                            'label': "%04d-label.png"%name}
    #                 save_image(data[i], os.path.join(result_path, fileset['data']),normalize=True)
    #                 save_image(label[i], os.path.join(result_path, fileset['label']),normalize=True)
    #                 save_image(output[i], os.path.join(result_path, fileset['output']),normalize=True)

                

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    model = pix2pix(args)
    
    model.train(ckpt_path=args.ckpt_path, result_path=args.result_path)

    # model.test(ckpt_path=args.ckpt_path, result_path=args.result_path)
 







                #    sys.stdout.write(
                #     "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                #     % (
                #         epoch,
                #         opt.n_epochs,
                #         i,
                #         len(dataloader),
                #         loss_D.item(),
                #         loss_G.item(),
                #         loss_pixel.item(),
                #         loss_GAN.item(),
                #         time_left,
                #     )
                # )
