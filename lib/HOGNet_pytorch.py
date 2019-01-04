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
        self.G_f = self.f1 = self.bf = None

    def _comp_mask(self, mask):
        BS = self.BS
        # print('COMPILING')
        t = time()     
        mask = torch.Tensor(mask)  
        bf = torch.ones([1, 1, 2 * BS, 2 * BS])
        m_b = F.conv2d(mask, bf, padding=(BS // 2, BS // 2), stride=BS)
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

        x = (x_o + 1) / 2
        if self.G_f is None:
            Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
            Gy = Gx.T
            f1_w = []
            for i in range(self.NO):
                t = np.pi / self.NO * i
                g = np.cos(t) * Gx + np.sin(t) * Gy
                gg = np.tile(g[np.newaxis, np.newaxis, :, :], [1, 1, 1, 1])
                f1_w.append(gg)
            f1_w = np.concatenate(f1_w, axis=0)
            G = np.concatenate([Gx[np.newaxis, np.newaxis, :, :], Gy[np.newaxis, np.newaxis, :, :]], axis=0)
            self.G_f = torch.Tensor(G).cuda()
            a = np.cos(np.pi / self.NO)
            self.l1 = torch.Tensor([1 / (1 - a)]).cuda()
            self.l2 = torch.Tensor([a / (1 - a)]).cuda()
            eps = 1e-3
            if self.nc == 3:
                x_gray = x.mean(1).unsqueeze(1)
            else:
                x_gray = x
            self.f1 = torch.Tensor(f1_w).cuda()
        h0 = F.conv2d(x_gray, self.f1, padding=(1,1)).abs()
        g = F.conv2d(x_gray, self.G_f, padding=(1,1))

        if self.use_bin:
            gx = g[:, [0], ...]
            gy = g[:, [1], ...]
            gg = torch.sqrt(gx * gx + gy * gy + eps)
            hk = self.l1 * h0 - self.l2 * gg
            hk[hk <= 0] = 0

            if self.bf is None:
                bf_w = torch.zeros((self.NO, self.NO, 2 * self.BS, 2 * self.BS))
                b = 1 - np.abs((np.arange(1, 2 * self.BS + 1) - (2 * self.BS + 1.0) / 2.0) / self.BS)
                b = b[np.newaxis, :]
                bb = b.T.dot(b)
                bb = torch.Tensor(bb)
                for n in range(self.NO):
                    bf_w[n, n] = bb
                self.bf = bf_w.cuda()
            h_f = F.conv2d(hk, self.bf, padding=(self.BS // 2, self.BS // 2), stride=self.BS)
            return h_f
        else:
            return g
