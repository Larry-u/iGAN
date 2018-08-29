import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from time import time
# NO = 8
# BS = 8


class HOGNet():
    def __init__(self, use_bin=True, NO=8, BS=8, nc=3):
        self.use_bin = True
        self.NO = NO
        self.BS = BS
        self.nc = nc
        self.use_bin = use_bin

    def _comp_mask(self, mask):
        BS = self.BS
        # print('COMPILING')
        t = time()     
        mask = torch.Tensor(mask)  
        bf = torch.ones([1, 1, 2 * BS, 2 * BS])
        m_b = F.conv2d(mask, bf, padding=(BS / 2, BS / 2), stride=BS)
        return m_b

    def comp_mask(self, _masks):
        _masks = self._comp_mask(_masks)# > 1e-5        
        masks = torch.zeros_like(_masks)
        masks[_masks > 1e-5] = 1
        return masks
    def get_hog(self, x_o):
        if isinstance(x_o, np.ndarray):
            x_o = torch.Tensor(x_o)
        if isinstance(x_o, torch.Tensor):
            x_o = x_o.cuda()
        use_bin = self.use_bin
        NO = self.NO
        BS = self.BS
        nc = self.nc
        x = (x_o + 1) / 2
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
        Gy = Gx.T
        f1_w = []
        for i in range(NO):
            t = np.pi / NO * i
            g = np.cos(t) * Gx + np.sin(t) * Gy
            gg = np.tile(g[np.newaxis, np.newaxis, :, :], [1, 1, 1, 1])
            f1_w.append(gg)
        f1_w = np.concatenate(f1_w, axis=0)
        G = np.concatenate([Gx[np.newaxis, np.newaxis, :, :], Gy[np.newaxis, np.newaxis, :, :]], axis=0)
        G_f = torch.Tensor(G).cuda()

        a = np.cos(np.pi / NO)
        l1 = 1 / (1 - a)
        l2 = a / (1 - a)
        eps = 1e-3
        if nc == 3:
            x_gray = x.mean(1).unsqueeze(1)
        else:
            x_gray = x
        f1 = torch.Tensor(f1_w).cuda()
        h0 = F.conv2d(x_gray, f1, padding=(1,1)).abs()
        g = F.conv2d(x_gray, G_f, padding=(1,1))

        if use_bin:
            gx = g[:, [0], ...]
            gy = g[:, [1], ...]
            gg = torch.sqrt(gx * gx + gy * gy + eps)
            hk = torch.Tensor([l1]).cuda() * h0 - torch.Tensor([l2]).cuda() * gg
            hk[hk <= 0] = 0

            bf_w = torch.zeros((NO, NO, 2 * BS, 2 * BS))
            b = 1 - np.abs((np.arange(1, 2 * BS + 1) - (2 * BS + 1.0) / 2.0) / BS)
            b = b[np.newaxis, :]
            bb = b.T.dot(b)
            bb = torch.Tensor(bb)
            for n in range(NO):
                bf_w[n, n] = bb

            bf = bf_w.cuda()
            h_f = F.conv2d(hk, bf, padding=(BS / 2, BS / 2), stride=BS)
            return h_f
        else:
            return g
