#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models.

"""

import os
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import sys
import torch.nn.functional as F
from itertools import chain
import numpy as np
from torch.distributions import Normal
from scipy import special as sp
from random import sample
import numpy as np

#project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#sys.path.append(project_path)

#from Utils.TopicPlayer import Topic, TopicTrends
from Utils.xxmax import sparsemax, entmax15

def sample_orthonormal_to(mu, dim):
    """Sample point on sphere orthogonal to mu.
    """
    v = torch.randn(dim)
    rescale_value = mu.dot(v) / mu.norm()
    proj_mu_v = mu * rescale_value.expand(dim)
    ortho = v - proj_mu_v
    ortho_norm = torch.norm(ortho)
    return ortho / ortho_norm.expand_as(ortho)

def sample_vmf_w(kappa, m):

    b = (-2 * kappa + np.sqrt(4. * kappa ** 2 + (m - 1) ** 2)) / (m - 1)
    a = (m - 1 + 2 * kappa + np.sqrt(4 * kappa ** 2 + (m - 1) ** 2)) / 4
    d = 4 * a * b / (1 + b) - (m - 1) * np.log(m - 1)
    while True:
        z = np.random.beta(0.5 * (m - 1), 0.5 * (m - 1))
        W = (1 - (1 + b) * z) / (1 + (1 - b) * z)
        T = 2 * a * b / (1 + (1 - b) * z)
        u = np.random.uniform(0, 1)
        if (m - 1) * np.log(T) - T + d >= np.log(u):
            return W
        
def vmf_sampler(mu, kappa):
    mu = mu.cpu()
    batch_size, id_dim = mu.size()
    result_list = []
    for i in range(batch_size):
        munorm = mu[i].norm().expand(id_dim)  # TODO norm p=?
        if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
            # sample offset from center (on sphere) with spread kappa
            # w = self._sample_weight(self.kappa, id_dim)     # TODO mine?

            w = sample_vmf_w(kappa, id_dim)

            wtorch = w * torch.ones(id_dim)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = sample_orthonormal_to(mu[i] / munorm, id_dim)
            # v= vMF.sample_vmf_v(mu[i])

            # compute new point
            scale_factr = torch.sqrt(torch.ones(id_dim)) - torch.pow(wtorch, 2)
            orth_term = v * scale_factr

            muscale = mu[i] * wtorch / munorm
            sampled_vec = (orth_term + muscale) * munorm
        else:
            rand_draw = torch.randn(id_dim)
            rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
            rand_norms = (torch.rand(1)).expand(id_dim)
            sampled_vec = rand_draw * rand_norms  # mu[i]
        result_list.append(sampled_vec)

    return torch.stack(result_list, 0)

def removeNAN(tensor, replaceNaN, replaceInf):
    tensor[torch.isnan(tensor)] = replaceNaN
    tensor[torch.isinf(tensor)] = replaceInf
    return tensor

def eps_from(mean, std, size):
    t = torch.randn(size)
    #gaussian = Normal(0, 1)
    #gaussian.sample()
    try:
        res =  mean + t.mul(std)
    except:
        print('eeeeeeerror')
        print(t.size())
        print(size[0],size[1])
        exit()
    return res

#def norm_tensor(tensor):
#    for t in range(tensor.size(0)):
#        total = torch.sum(tensor[t])
#        tensor[t] = tensor[t]/float(total)
#    return tensor
    
def sparse(weight_tensor_data, p=0.0, dim=0):
    if p == 0.0:
        return weight_tensor_data
    else:
        if dim==0:
            n_row, n_col = weight_tensor_data.size()
            n_zeros = int(n_col*p)
            for row in weight_tensor_data:
                z_idx = sample(range(0,n_col), n_zeros)
                for idx in z_idx:
                    row[z_idx] = 0.0
            return weight_tensor_data
        elif dim==-1:
            n_row, n_col = weight_tensor_data.size()
            n_zeros = int(n_row*p)
            for col in range(n_col):
                z_idx = sample(range(0,n_row), n_zeros)
                for idx in z_idx:
                    weight_tensor_data[z_idx][col] = 0.0
            return weight_tensor_data            
        
    


class VAEencoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sparseW=0.0, entMax=False):
        super(VAEencoder, self).__init__()
        self.numTopic = z_dim
        self.fcEn1 = nn.Linear(x_dim, h_dim)        
        self.fcMiu = nn.Linear(h_dim, z_dim)      
        self.meanBN = nn.BatchNorm1d(z_dim)                  
        self.fcSigma = nn.Linear(h_dim, z_dim)         
        self.sigmaBN = nn.BatchNorm1d(z_dim)  
        self.entMax_ = entMax
        self.sparseW = sparseW
        self.sparseWeight()
    
    def sparseWeight(self):
        # sparse weights
        self.fcEn1.weight.data = sparse(self.fcEn1.weight.data, self.sparseW)
        
        
        
    def forward(self, data, bn=False):
#        print('encoder en1size', self.fcEn1.weight.data.size())
#        print('data ', data.size())
        en1 = self.fcEn1(data)
#        print('en1 size, ', en1.size())
        if self.entMax_:
            en1 = entmax15(en1, dim=-1)
        else:
            en1 = F.softmax(en1, dim=-1)   
#        en1 = F.softplus(en1)
#        print('en1 size, ', en1.size())
             
        if bn:                 
            miu   = self.meanBN(self.fcMiu(en1))       
            sigma = self.sigmaBN(self.fcSigma(en1))   
        else:
            miu   = self.fcMiu(en1)     
            sigma = self.fcSigma(en1)  
        #posterior_var = torch.exp(sigma)
        return miu, sigma
    
    
#    def paras(self):
#        return [self.fcEn1.parameters(), self.fcMiu, self.fcSigma]
        
class VAEdecoder(nn.Module):
    def __init__(self, z_dim, x_dim, entMax=False, sparseW=0.0):
        super(VAEdecoder, self).__init__()
        self.fcG1 = nn.Linear(z_dim, x_dim)
        self.relu = nn.ReLU()
        self.entMax_ = entMax
        self.sparseW = sparseW
        
    def sparseWeight(self):
        self.fcG1.weight.data = sparse(self.fcG1.weight.data, self.sparseW, dim=-1)
        
    def forward(self, z):
        recon = self.fcG1(z)
        #recon = self.relu(recon)
        
        if self.entMax_:
            #recon = entMax(recon, dim=-1)
            recon = entmax15(recon, dim=-1)
        else:
            recon = F.softmax(recon, dim=-1)
        #recon = nn.ReLU(recon)
        #recon = sparse(recon, 0.9)
        return recon
    
#    def paras(self):
#        return [self.fcG1]

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sampler, kappa=0, useGPU=False, prior_mu=0.0, prior_sigma=1.0,
                 sparseW=False, entMax=False, bn=False):
        super(VAE, self).__init__()
        self.useGPU = useGPU
        self.z_dim = z_dim
        self.enc = VAEencoder(x_dim, h_dim, z_dim, sparseW)
        self.dec = VAEdecoder(z_dim, x_dim, entMax, sparseW)
        self.sampler = sampler
        self.kappa = kappa
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.entMax_ = entMax
        
        self.post_mu = 0.0
        self.post_sigma = 0.0
        self.post_mu_n = 0 
        self.post_sigma_n = 0
        
        self.bn = bn
        
    def init_weight(self, embedding=None):
        self.enc.fcEn1.weight.data = embedding.transpose(0,1)
        
    def convey(self, tw_data=None, pre_mu=None, pre_sigma=None, alpha=0.7, beta=0.3):
        self.enc.fcMiu.weight.data = torch.from_numpy(pre_mu).mul(alpha) + (self.enc.fcMiu.weight.data.clone().mul(beta))
        self.enc.fcSigma.weight.data = torch.from_numpy(pre_sigma).mul(alpha) + (self.enc.fcSigma.weight.data.clone().mul(beta))
        if tw_data is not None:
            self.dec.fcG1.weight.data = torch.from_numpy(tw_data).mul(alpha) + (self.dec.fcG1.weight.data.clone().mul(beta))
        print('convey data')
        
    def update_post(self, m, s):
        #assert m.size(0) == s.size(0)
#        print('m, s')
#        print(m)
#        print('----')
#        print(s)
        batch_n = m.size(0) * m.size(1)
        m = float(torch.sum(m) / batch_n)
        s = float(torch.sum(s) / batch_n)
        temp_m = self.post_mu * self.post_mu_n + m
        temp_s = self.post_sigma * self.post_sigma_n + s
        self.post_mu_n += 1
        self.post_sigma_n += 1
        self.post_mu += temp_m / self.post_mu_n
        self.post_sigma += temp_s / self.post_sigma_n
#        print(self.post_mu, self.post_sigma)
    def forward(self, data):
#        print('input data size ', data.size())
#        print('enc size', self.enc.fcEn1.weight.data.size())
        
        
        mu, sigma = self.enc(data, self.bn)
        #self.update_post(mu.data.clone().cpu(), sigma.data.clone().cpu())
#        print('shape of mu and sigma ', mu.size(), sigma.size())
        z = self.sample(mu=mu, sigma=sigma)
        if self.entMax_:
            z = entmax15(z, dim=-1)
        else:
            z = F.softmax(z, dim=-1)
#        print('z size, ', z.size(0))
        recon = self.dec(z)
        kld = self._kld(mu, sigma)
        rec_loss = self._recon_loss(data, recon)
        #topic_loss = self._tloss()
#        print('shape kld: ', kld.size())
#        print('rec_loss shape: ', rec_loss.size()) # one float
        loss = kld - rec_loss #+ topic_loss
        #loss = self.loss(data, recon)
        return recon, kld, rec_loss, loss
        
    def sample(self, data=None, mu=None, sigma=None, bn=True):
        if data is not None:
            mu, sigma = self.enc(data, bn)
        
        # Gaussian
        if self.sampler == 'gaussian':
            eps1 = eps_from(self.prior_mu, self.prior_sigma, sigma.size())
            eps1 = eps1.type_as(sigma)
            eps1 = Variable(eps1)
            if self.useGPU:
                eps1 = eps1.cuda()
            z = mu + sigma.mul(eps1)
        
        # vMF
        elif self.sampler == 'vmf':
            tm = torch.tensor([self.prior_mu])
            tm = tm.expand(mu.size())
            if self.useGPU:
                tm = tm.cuda()
            eps1 = vmf_sampler(tm, self.kappa) # prior ()
            eps2 = vmf_sampler(mu, self.kappa) # posterior 
            if self.useGPU:
                eps1 = eps1.cuda()
                eps2 = eps2.cuda()
            z1 = mu + sigma.mul(eps1)
            z2 = mu + sigma.mul(eps2)
            z = (z1+z2)/2

        return z
    
    def decoding(self, z):
        return self.dec(z)
    
    def _kld(self, mu2, sigma2):
        mean_sq = (mu2-self.prior_mu).mul((mu2-self.prior_mu))
        std_sq = (sigma2-self.prior_sigma).mul(sigma2-self.prior_sigma)
        kld = 0.5*( mean_sq + std_sq - 1 - torch.log(std_sq))
        return torch.mean(kld)
    
    def _recon_loss(self, data, data_hat):
        #bce = F.binary_cross_entropy(data_hat, data)
        l = torch.nn.MSELoss()
        rec_l = l(data, data_hat)
        return rec_l
    
    def _tloss(self):
        tw = self.dec.fcG1.weight.data.clone()
        twt = tw.transpose(0,1)
        tprod = tw.mm(twt)
        temp = tprod.clone()
        matrix_i = torch.zeros(temp.size())
        if self.useGPU:
            matrix_i = matrix_i.cuda()
        for i in range(matrix_i.size(0)):
            matrix_i[i][i] = 1.0
        tloss = tprod - matrix_i
        tloss = torch.det(tloss)
        return tloss
        
    
    
    def get_tw(self, trans=True):
        if trans:
            return self.dec.fcG1.weight.data.clone().transpose(0,1)
        else:
            a = self.dec.fcG1.weight.data.clone().cpu().numpy()
            b = self.enc.fcMiu.weight.data.clone().cpu().numpy()
            c = self.enc.fcSigma.weight.data.clone().cpu().numpy()
            return a,b,c
    
#    def paras(self):
#        return self.enc.paras() + self.dec.paras()

def vmfKL(k, d):
    return k * ((sp.iv(d / 2.0 + 1.0, k) \
                 + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
           + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
           - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2


  
'''    
class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, useGPU=True):
        super(VRNN, self).__init__()
        
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.useGPU = useGPU
        
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())
        
        #encoder
        self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())
        
        #prior
        self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.prior_mean = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())
        self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())
        
        #decoder
        self.dec = nn.Sequential(
			nn.Linear(z_dim, z_dim),
			nn.ReLU())
#        self.dec_std = nn.Sequential(
#			nn.Linear(h_dim, x_dim),
#			nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
			nn.Linear(z_dim, x_dim),
			nn.Sigmoid())
        
        #recurrence
        self.rnn = nn.Sequential(
			nn.Linear(z_dim + h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
    
        #recurrence
        #self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
        
        
    def forward(self, x, h):
        
        #h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
        phi_x_t = self.phi_x(x)
        
        #encoder
        enc_t = self.enc(torch.cat([phi_x_t, h.expand(phi_x_t.size())], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)
        
        #prior
        prior_t = self.prior(h)   ##
        prior_mean_t = self.prior_mean(prior_t)  ##
        prior_std_t = self.prior_std(prior_t)    ##
        
        #sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)   ##
        #phi_z_t = self.phi_z(z_t)    ##
        
        #decoder
        dec_t = self.dec(z_t)
        recon = self.dec_mean(dec_t)
        #dec_std_t = self.dec_std(dec_t)
        
        next_h = self.rnn(torch.cat((z_t, phi_x_t, h.expand(phi_x_t.size())), -1))
        
        #computing losses
        kld_loss = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
        #nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
        recon_loss = self._recon_loss(x , recon)
        
        loss = recon_loss - kld_loss
        
        return next_h, kld_loss.sum(), recon_loss.sum(), loss.sum()
    
    def get_tw(self):
        return self.dec_mean[0].weight.data.transpose(0,1)
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
            
    def _init_weights(self, stdv):
        pass
    
    
    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        if self.useGPU:
            eps = eps.cuda()
        return eps.mul(std).add_(mean)
    
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
   
#        var_division = torch.abs(sigma2/sigma1)
#        mu_differ = mu1 - mu2
#        kld = torch.log(var_division) -0.5 + 0.5*((
#                sigma1**2+mu_differ.mul(mu_differ))/sigma2**2)   
#        kld = kld.sum(-1)
     
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element, 1)
    
    
    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))
    
    
    def _nll_gauss(self, mean, std, x):
        pass
    
    def _recon_loss(self, x, x_recon):
        return -torch.sum(x*((x_recon+1e-10).log()), 1)
'''    
#class VRNN_vmf(nn.Module):
#    def __init__(self, x_dim, h_dim, z_dim, n_layers, k, sphere_dim, bias=False, useGPU=True):
#        super(VRNN_vmf, self).__init__()
#        
#        self.kappa = k
#        self.sphere_dim = sphere_dim
#        self.kld = self._kld_vmf(k, sphere_dim)
#        
#        self.x_dim = x_dim
#        self.h_dim = h_dim
#        self.z_dim = z_dim
#        self.n_layers = n_layers
#        self.useGPU = useGPU
#        
#        #feature-extracting transformations
#        self.phi_x = nn.Sequential(
#			nn.Linear(x_dim, h_dim),
#			nn.ReLU(),
#			nn.Linear(h_dim, h_dim),
#			nn.ReLU())
#        self.phi_z = nn.Sequential(
#			nn.Linear(z_dim, h_dim),
#			nn.ReLU())
#        
#        #encoder
#        self.enc = nn.Sequential(
#			nn.Linear(h_dim + h_dim, h_dim),
#			nn.ReLU(),
#			nn.Linear(h_dim, h_dim),
#			nn.ReLU())
#        self.enc_mean = nn.Linear(h_dim, z_dim)
#        self.enc_std = nn.Sequential(
#			nn.Linear(h_dim, z_dim),
#			nn.Softplus())
#        
#        #prior
#        self.prior = nn.Sequential(
#			nn.Linear(h_dim, h_dim),
#			nn.ReLU())
#        self.prior_mean = nn.Sequential(
#			nn.Linear(h_dim, z_dim),
#			nn.Softplus())
##        self.prior_std = nn.Sequential(
##			nn.Linear(h_dim, z_dim),
##			nn.Softplus())
#        
#        #decoder
#        self.dec = nn.Sequential(
#			nn.Linear(z_dim, z_dim),
#			nn.ReLU())
##        self.dec_std = nn.Sequential(
##			nn.Linear(h_dim, x_dim),
##			nn.Softplus())
#        #self.dec_mean = nn.Linear(h_dim, x_dim)
#        self.dec_mean = nn.Sequential(
#			nn.Linear(z_dim, x_dim),
#			nn.Sigmoid())
#        
#        #recurrence
#        self.rnn = nn.Sequential(
#			nn.Linear(z_dim + h_dim + h_dim, h_dim),
#			nn.ReLU(),
#			nn.Linear(h_dim, h_dim),
#			nn.ReLU())
#    
#        #recurrence
#        #self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
#        
#        
#    def forward(self, x, h, test=False):
#        
#        #h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
#        phi_x_t = self.phi_x(x)
#        
#        #encoder
#        enc_t = self.enc(torch.cat([phi_x_t, h.expand(phi_x_t.size())], 1))
#        enc_mean_t = self.enc_mean(enc_t)
#        enc_std_t = self.enc_std(enc_t)
#        
#        #prior
#        prior_t = self.prior(h)   ##
#        prior_mean_t = self.prior_mean(prior_t)  ##
#        #prior_t = vmf_sampler(prior_mean_t, 0)
#        #prior_std_t = self.prior_std(prior_t)    ##
#        
#        #sampling and reparameterization
#        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t, prior_mean_t)   ##
#        if test == True:
#            return z_t.detach().data
#        #phi_z_t = self.phi_z(z_t)    ##
#        
#        #decoder
#        dec_t = self.dec(z_t)
#        recon = self.dec_mean(dec_t)
#        #dec_std_t = self.dec_std(dec_t)
#        
#        next_h = self.rnn(torch.cat((z_t, phi_x_t, h.expand(phi_x_t.size())), -1))
#        
#
#        #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
#        #nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
#        recon_loss = self._recon_loss(x , recon)
#        
#        #computing losses
#        kld_loss = self.kld
#        if self.useGPU:
#            kld_loss = kld_loss.cuda()
#        #self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
#        
#        loss = recon_loss - kld_loss
#        
#        return next_h, kld_loss*x.size(0), recon_loss.sum(), loss.sum()
#    
#    def get_tw(self):
#        return self.dec_mean[0].weight.data.clone().transpose(0,1)
#    
##    def reset_parameters(self, stdv=1e-1):
##        for weight in self.parameters():
##            weight.data.normal_(0, stdv)
#    def reset_parameters(self, tw=None):
#        if tw:
#            try:
#                self.dec_mean[0].weight.data = tw
#                print('reset')
#            except:
#                for weight in self.parameters():
#                    weight.data.normal_(0, 1e-1)                
#        else:
#            for weight in self.parameters():
#                weight.data.normal_(0, 1e-1)
#            
#    def _init_weights(self, stdv):
#        pass
#    
#    
#    def _reparameterized_sample(self, mean, std, prior_mean):
#        """using std to sample"""
#        #eps = torch.FloatTensor(std.size()).normal_()
#        eps = vmf_sampler(prior_mean, 0)
#        #eps = vmf_sampler(mean, self.kappa)
#        eps = Variable(eps)
#        if self.useGPU:
#            eps = eps.cuda()
#        return eps.mul(std).add_(mean)
#    
#    
#    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
#        """Using std to compute KLD"""
#   
##        var_division = torch.abs(sigma2/sigma1)
##        mu_differ = mu1 - mu2
##        kld = torch.log(var_division) -0.5 + 0.5*((
##                sigma1**2+mu_differ.mul(mu_differ))/sigma2**2)   
##        kld = kld.sum(-1)
#     
#        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
#			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
#			std_2.pow(2) - 1)
#        return	0.5 * torch.sum(kld_element)
#    
#    def _kld_vmf(self, k, d):
#        kld = k * ((sp.iv(d / 2.0 + 1.0, k) \
#                     + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
#               + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
#               - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2
#        return torch.tensor(kld)
#
#    def _nll_bernoulli(self, theta, x):
#        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))
#    
#    
#    def _nll_gauss(self, mean, std, x):
#        pass
#    
#    def _recon_loss(self, x, x_recon):
#        return -torch.sum(x*((x_recon+1e-10).log()), 1)