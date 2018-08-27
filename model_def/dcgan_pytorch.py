from time import time
import numpy as np
import torch
from HCIGAN.model import Generator

class Model(object):
    def __init__(self):
        self.nz = 512  # [hack] hard-coded
        self._gen = Generator()


    def model_G(self, z):  # generative model z => x
        return self._gen(z)

    def gen_samples(self, z0=None, n=32, batch_size=32, use_transform=True):
        assert n % batch_size == 0

        samples = []

        if z0 is None:
            z0 = np_rng.uniform(-1., 1., size=(n, self.nz))
        else:
            n = len(z0)
            batch_size = max(n, 64)
        n_batches = int(np.ceil(n / float(batch_size)))
        for i in range(n_batches):
            zmb = floatX(z0[batch_size * i:min(n, batch_size * (i + 1)), :])
            xmb = self._gen(zmb)
            samples.append(xmb)

        samples = np.concatenate(samples, axis=0)
        if use_transform:
            samples = self.inverse_transform(samples, npx=self.npx, nc=self.nc)
            samples = (samples * 255).astype(np.uint8)
        return samples

    def transform(self, x, nc=3):
        if nc == 3:
            return floatX(x).transpose(0, 3, 1, 2) / 127.5 - 1.
        else:
            return floatX(x).transpose(0, 3, 1, 2) / 255.0

    def transform_mask(self, x):
        return x.permute(0, 3, 1, 2) / 255.0

    def inverse_transform(self, x, npx=64, nc=3):
        if nc == 3:
            return (x.reshape(-1, 3, npx, npx).transpose(0, 2, 3, 1) + 1.) / 2.
        else:
            return 1.0 - x.reshape(-1, 1, npx, npx).transpose(0, 2, 3, 1)


