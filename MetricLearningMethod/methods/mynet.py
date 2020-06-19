# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from torch.nn import Parameter

import utils

class ProtoNet_Asoftmax(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet_Asoftmax, self).__init__( model_func,  n_way, n_support)#call MetaTemplate
        self.loss_fn  = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)#(5,5,512)  and (5,16,512)

        z_support   = z_support.contiguous()#copy whole memory

        #(5,5,512)
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim] (mean of features)
        #z_proto.shape=(5,512)
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )#[75, n_dim]
        #(80,512)


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))#[75] #[80] n_way=5 n_query=16
        #[0,0,...,0, 1,1,...,1 .....,4,4,4,4...,4]
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss
    def load_state_dict(self, state_dict):
        #https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state: continue
            if isinstance(param, Parameter): param = param.data# backwards compatibility for serialized parameters
            own_state[name].copy_(param)

def euclidean_dist( x, y):
    # x: N x D => 80 x 512
    # y: M x D => 5 x 512
    n = x.size(0)#80
    m = y.size(0)#5
    d = x.size(1)#512
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)#(80,5,512)
    y = y.unsqueeze(0).expand(n, m, d)#(80,5,512)

    return torch.pow(x - y, 2).sum(2)#(80,5) cue 5-way 16 qeurys => 80 images in total correntsponding to 5 labels
