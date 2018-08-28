from time import time
import numpy as np
import torch
from HCIGAN.model import Generator
from torch import distributions as dist
class Model(object):
    def __init__(self, model_name, model_file):
        self.model_name = model_name
        self.nz = 512  # [hack] hard-coded
        self._gen = Generator()
        generator_path = '/home/toby/Documents/HCIGAN/data/network-final.pth'
        self._gen.load_state_dict(torch.load(generator_path))
        self.npx = 1024 # width = height = 1024px
        self.nc = 3 # 3 channels

    def model_G(self, z):  # generative model z => x
        return self._gen(z, img=True)

    def gen_samples(self, z0=None, n=32, batch_size=32, use_transform=True):
        assert n % batch_size == 0

        samples = []

        if z0 is None:
            z0 = torch.FloatTensor(n, self.nz).uniform_(-1, 1) #2 * torch.rand(n, self.nz) - 1
        else:
            n = len(z0)
            batch_size = max(n, 64)
        n_batches = int(np.ceil(n / float(batch_size)))
        for i in range(n_batches):
            zmb = torch.Tensor(z0[batch_size * i:min(n, batch_size * (i + 1)), :])
            xmb = self._gen(zmb, img=use_transform)
            samples.append(xmb)

        samples = np.concatenate(samples, axis=0)
        return samples

    def transform(self, x, nc=3):
        if nc == 3:
            return torch.Tensor(x).permute(0, 3, 1, 2) / 127.5 - 1. #range -1, 1
        else:
            return torch.Tensor(x).permute(0, 3, 1, 2) / 255.0 # range 0, 1

    def transform_mask(self, x):
        return x.permute(0, 3, 1, 2) / 255.0

    def inverse_transform(self, x, npx=64, nc=3):
        if nc == 3:
            return (x.view(-1, 3, npx, npx).permute(0, 2, 3, 1) + 1.) / 2.
        else:
            return 1.0 - x.reshape(-1, 1, npx, npx).transpose(0, 2, 3, 1)


