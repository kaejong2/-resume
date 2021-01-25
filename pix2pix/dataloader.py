import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

class data_loader(torch.utils.data.Dataset):
    def __init__(self, data_dir, nch=3, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.nch = nch
        self.data_list = os.listdir(data_dir)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.ToTensor()])

    def __getitem__(self, index):
        data = plt.imread(os.path.join(self.data_dir, self.data_list[index]))[:, :, :self.nch]

        if data.dtype == np.uint8:
            data = data / 255.0

        half = int(data.shape[1]/2)

        A = data[:, :half, :]
        B = data[:, half:, :]

        data = {'input': A, 'output': B}
        

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data_list)