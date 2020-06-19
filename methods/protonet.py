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
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists#negation cuz we want fetch largest similarity as the small distance
        #import pdb
        #pdb.set_trace()

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)#3
    m = y.size(0)#4
    d = x.size(1)#5
    assert d == y.size(1)

    #x=tensor([[0.5406, 0.3365, 0.1761, 0.2583, 0.8337],
    #    [0.7021, 0.3330, 0.8095, 0.9374, 0.8500],
    #    [0.1519, 0.3289, 0.7063, 0.6907, 0.0821]])


    #y=tensor([[0.2957, 0.2381, 0.4905, 0.2698, 0.6617],
    #    [0.8519, 0.6391, 0.1079, 0.3420, 0.3543],
    #    [0.6536, 0.4376, 0.1066, 0.8189, 0.8344],
    #    [0.3118, 0.2604, 0.1548, 0.1503, 0.1528]])

    x = x.unsqueeze(1).expand(n, m, d)#unsqueeze=(3,"1",5) then expand it becomes (3,4,5)
    #tensor([[[0.5406, 0.3365, 0.1761, 0.2583, 0.8337],
    #     [0.5406, 0.3365, 0.1761, 0.2583, 0.8337],
    #     [0.5406, 0.3365, 0.1761, 0.2583, 0.8337],
    #     [0.5406, 0.3365, 0.1761, 0.2583, 0.8337]],
    #
    #    [[0.7021, 0.3330, 0.8095, 0.9374, 0.8500],
    #     [0.7021, 0.3330, 0.8095, 0.9374, 0.8500],
    #     [0.7021, 0.3330, 0.8095, 0.9374, 0.8500],
    #     [0.7021, 0.3330, 0.8095, 0.9374, 0.8500]],
    #
    #    [[0.1519, 0.3289, 0.7063, 0.6907, 0.0821],
    #     [0.1519, 0.3289, 0.7063, 0.6907, 0.0821],
    #     [0.1519, 0.3289, 0.7063, 0.6907, 0.0821],
    #     [0.1519, 0.3289, 0.7063, 0.6907, 0.0821]]])
    y = y.unsqueeze(0).expand(n, m, d)#unsqueeze=("1",4,5) then expand it becomes (3,4,5)
    #tensor([[[0.2957, 0.2381, 0.4905, 0.2698, 0.6617],
    #     [0.8519, 0.6391, 0.1079, 0.3420, 0.3543],
    #     [0.6536, 0.4376, 0.1066, 0.8189, 0.8344],
    #     [0.3118, 0.2604, 0.1548, 0.1503, 0.1528]],
    #
    #    [[0.2957, 0.2381, 0.4905, 0.2698, 0.6617],
    #     [0.8519, 0.6391, 0.1079, 0.3420, 0.3543],
    #     [0.6536, 0.4376, 0.1066, 0.8189, 0.8344],
    #     [0.3118, 0.2604, 0.1548, 0.1503, 0.1528]],
    #
    #    [[0.2957, 0.2381, 0.4905, 0.2698, 0.6617],
    #     [0.8519, 0.6391, 0.1079, 0.3420, 0.3543],
    #     [0.6536, 0.4376, 0.1066, 0.8189, 0.8344],
    #     [0.3118, 0.2604, 0.1548, 0.1503, 0.1528]]])

    return torch.pow(x - y, 2).sum(2)#shape(3,4)
