
import glob
import random
import os
from arguments import Arguments
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

args = Arguments().parser().parse_args()


def data_loader(args):
    # Dataset loader
    transforms_ = [ transforms.Resize(int(args.image_size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(args.image_size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(args.root_path, transforms_=transforms_, unaligned=True), 
                            batch_size=args.batch_size, shuffle=True)

    return dataloader

# print(data_loader(args))
# data = data_loader(args)
# for i, data in enumerate(data):
#     print(data["A"])
#     print(data["B"])

