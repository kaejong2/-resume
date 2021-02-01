import numpy as np
import torch
from PIL import Image

import os
import torchvision.transforms as transforms

class data_loader(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = os.listdir(data_dir)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.data_list[index])).convert("RGB")
        w, h = img.size

        A = img.crop((0, 0, w/2, h))
        B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            A = Image.fromarray(np.array(A)[:,::-1, :], "RGB")
            B = Image.fromarray(np.array(B)[:,::-1, :], "RGB")

        A = self.transform(A)
        B = self.transform(B)

        
        return B, A

    def __len__(self):
        return len(self.data_list)