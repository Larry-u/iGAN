import torch
from time import time
from lib import HOGNet_pytorch
import numpy as np
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
class OPT_Solver():
    def __init__(self, model, batch_size=8, d_weight=0.0):
        self.model = model
        self.npx = model.npx
        self.nc = model.nc
        self.nz = model.nz
        self.model_name = model.model_name
        self.transform = model.transform
        self.transform_mask = model.transform_mask
        self.inverse_transform = model.inverse_transform
        BS = 4 if self.nc == 1 else 8  # [hack]
        self.hog = HOGNet_pytorch.HOGNet(use_bin=True, NO=8, BS=BS, nc=self.nc)
        self.opt_model = self.def_invert(model, batch_size=batch_size, d_weight=d_weight, nc=self.nc)
        self.batch_size = batch_size

    def get_image_size(self):
        return self.npx

    def invert(self, constraints, z_i):
        [_invert, z, beta_r, z_const] = self.opt_model
        constraints_t = self.preprocess_constraints(constraints)
        [im_c_t, mask_c_t, im_e_t, mask_e_t] = constraints_t  # [im_c_t, mask_c_t, im_e_t, mask_e_t]

        results = _invert(im_c_t, mask_c_t, im_e_t, mask_e_t, z_i.astype(np.float32))

        [gx, cost, cost_all, rec_all, real_all, init_all, sum_e, sum_x_edge] = results
        with torch.no_grad():
            gx_t = self.model._gen(self.z_param, img= True)
            if self.nc == 1:
                gx_t = np.tile(gx_t, (1, 1, 1, 3))
            z_t = np.tanh(z.cpu().numpy()).copy()
        return gx_t, z_t, cost_all.detach().cpu().numpy()

    def preprocess_constraints(self, constraints):
        im_c_o, mask_c_o, im_e_o, mask_e_o = map(torch.Tensor,constraints)

        im_c = self.transform(im_c_o.unsqueeze(0), self.nc)
        mask_c = self.transform_mask(mask_c_o.unsqueeze(0))
        im_e = self.transform(im_e_o.unsqueeze(0), self.nc)
        mask_t = self.transform_mask(mask_e_o.unsqueeze(0))
        mask_e = self.hog.comp_mask(mask_t)
        shp = [self.batch_size, -1, -1, -1]
        im_c_t = im_c.expand(shp)
        mask_c_t = mask_c.expand(shp)
        im_e_t = im_e.expand(shp)
        mask_e_t = mask_e.expand(shp)
        return [im_c_t, mask_c_t, im_e_t, mask_e_t]

    def initialize(self, z0):
        self.z_param.data = torch.Tensor(np.arctanh(z0)).cuda()

    def set_smoothness(self, l):
        self.z_const = torch.Tensor([l]).cuda()

    def gen_samples(self, z0):
        samples = self.model.gen_samples(z0=z0)
        if self.nc == 1:
            samples = np.tile(samples, [1, 1, 1, 3])
        return samples

    def def_invert(self, model, batch_size=1, d_weight=0.0, nc=1, lr=0.1, b1=0.9, use_bin=True):
        assert d_weight == 0
        z = torch.FloatTensor(batch_size, self.nz).uniform_(-1, 1).cuda() #2 * torch.rand(n, self.nz) - 1
        print(f'z : {z.shape}')

        self.z_param = nn.Parameter(z)
        self.z_const = torch.Tensor([5]).cuda()
    
        self.adam = Adam([self.z_param], lr=lr, betas=(b1, 0.999))
        def _invert(x_c, m_c, x_e, m_e, z0):
            x_c = x_c.cuda()
            m_c = m_c.cuda()
            x_e = x_e.cuda()
            m_e = m_e.cuda()
            # print(f'm_e : {m_e}')
            z0  = torch.Tensor(z0).cuda()

            for i in range(1):
                self.adam.zero_grad()
                # print(f'self.z_param : {self.z_param.shape}')
                gx = model.model_G(self.z_param.cuda())
                _gx = torch.zeros_like(gx)
                for i in range(gx.shape[0]):
                    _gx[i] = gx[i] - gx[i].min() 
                    _gx[i] = gx[i] / gx[i].max() * 2 - 1 
                if nc == 1:  # gx, range [0, 1] => edge, 1
                    gx3 = 1.0 - _gx  # T.tile(gx, (1, 3, 1, 1))
                else:
                    gx3 = _gx
                mm_c = m_c.expand(-1, gx3.shape[1], -1, -1)
                # print(f'x_c.min(), x_c.max() : {x_c.min(), x_c.max()}')
                color_all = ((gx3 - x_c)**2 * mm_c).mean(-1).mean(-1).mean(-1) / (m_c.mean(-1).mean(-1).mean(-1) + 1e-5)
                gx_edge = self.hog.get_hog(gx3)
                x_edge = self.hog.get_hog(x_e)
                mm_e = m_e.expand(-1, gx_edge.shape[1], -1, -1)
                sum_e = mm_e.abs().sum()
                sum_x_edge = x_edge.abs().sum()
                edge_all = ((x_edge - gx_edge)**2 * mm_e).mean(-1).mean(-1).mean(-1) / (m_e.mean(-1).mean(-1).mean(-1) + 1e-5)
                rec_all = color_all + edge_all * 0.2
                init_all = ((z0 - self.z_param)**2).mean() * self.z_const
                cost_all = rec_all + init_all
                cost = cost_all.sum()
                cost.backward()
                self.adam.step()
            real_all = torch.zeros_like(cost_all)
            with torch.no_grad():
                return gx, cost, cost_all, rec_all, real_all, init_all, sum_e, sum_x_edge
        return [_invert, self.z_param, torch.Tensor([0]), self.z_const]
