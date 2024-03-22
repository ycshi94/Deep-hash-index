import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)

class JointEmbeder(nn.Module):

    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.hidden=config['n_hidden']
        self.hash_len=config['hash_len']
        self.code_encoder=nn.Linear(512, self.hash_len)
        self.desc_encoder = nn.Linear(512, self.hash_len)
        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        #self.init_weights()

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
    def code_encoding(self, x):
        hid = self.code_encoder(self.dropout(x))
        code = torch.tanh(self.alpha * hid)

        return code
        
    def desc_encoding(self, x):
        hid = self.desc_encoder(self.dropout(x))
        desc = torch.tanh(self.alpha * hid)

        return desc

    def hash_similarity(self, code_vec, desc_vec):
        s = torch.abs((code_vec - desc_vec) / 2)
        sum = torch.sum(s, dim=1)
        return sum
    def o_sim(self,code_vec, desc_vec):
        return torch.norm((code_vec-desc_vec)/2, p=1,dim=1)

    def forward(self, code_vec,desc_vec):
        Vc = F.normalize(code_vec)
        Vd = F.normalize(desc_vec)
        Sc = Vc.mm(Vc.t())
        Sd = Vd.mm(Vd.t())
        S = 0.6 * Sc + 0.4 * Sd
        SS = (0.6 * S + 0.4 / 128 * S.mm(S.t()))
        diag = torch.diag(SS)
        a_diag = torch.diag_embed(diag)
        II = torch.eye(len(a_diag))
        II=torch.Tensor(II).to('cuda')
        SS = SS - a_diag + II
        SS = 1.5 * SS
        one = torch.ones(len(a_diag), len(a_diag)).to('cuda')
        fin = torch.min(SS, one)
        code_hash=self.code_encoding(code_vec)
        desc_hash=self.desc_encoding(desc_vec)
        BcBd=code_hash.mm(desc_hash.t())
        BcBc=code_hash.mm(code_hash.t())
        BdBd=desc_hash.mm(desc_hash.t())
        l1=torch.norm(fin-BcBd/128)
        l2=torch.norm(fin-BcBc/128)
        l3=torch.norm(fin-BdBd/128)
        loss=l1+l2+l3
        return loss
