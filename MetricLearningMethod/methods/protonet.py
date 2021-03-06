# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)#call MetaTemplate
        self.loss_fn  = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()#copy whole memory
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim] (mean of features)
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )#[75, n_dim]


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))#[75]
        #[0,0,...,0, 1,1,...,1 .....,4,4,4,4...,4]
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
