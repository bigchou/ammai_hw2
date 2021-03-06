import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time, os, glob, pdb

import configs
import backbone

from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.myTPN import MyTPN

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import miniImageNet_few_shot

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        #if epoch % 10==0:
        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'Adam'

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":
            print("load miniImageNet [START]")
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            print("datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)")
            base_loader = datamgr.get_data_loader(aug = params.train_aug )#waste lots of time
            print("base_loader = datamgr.get_data_loader(aug = params.train_aug )")
            val_loader  = None
            print("load miniIMageNet [END]")
        else:
           raise ValueError('Unknown dataset')
        print("load model [START]")
        model           = BaselineTrain( model_dict[params.model], params.num_classes)
        print("load model [END]")

    elif params.method in ['protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":

            datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            base_loader        = datamgr.get_data_loader(aug = params.train_aug)
            val_datamgr        = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
            val_loader         = val_datamgr.get_data_loader(aug = False)

        else:
           raise ValueError('Unknown dataset')

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
    elif params.method in ['mytpn']:
        print("INIT mytpn")
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
        base_loader        = datamgr.get_data_loader(aug = params.train_aug)
        #the above line waste a lot of time
        val_datamgr        = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
        val_loader         = val_datamgr.get_data_loader(aug = False)
        
        
        
        model              = MyTPN( model_dict[params.model], **train_few_shot_params )
        #import pdb
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)#print(name, param.data)
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)#"./logs/checkpoints/miniImagenet/ResNet10_baseline"
    if params.train_aug:
        params.checkpoint_dir += '_aug'#"./logs/checkpoints/miniImagenet/ResNet10_baseline_aug"

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch#0
    stop_epoch = params.stop_epoch#400
    print("The checkpoint is saved to %s"%(params.checkpoint_dir))

    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
