import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import miniImageNet_few_shot, EuroSAT_few_shot, ISIC_few_shot


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def meta_test(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    #novel_loader has 600 dataloaders
    #n_query=15
    #pretrained_dataset=miniImageNet
    #freeze_backbone=True
    #n_way=5
    #n_support = 5
    correct = 0
    count = 0

    iter_num = len(novel_loader)#600

    acc_all = []

    for ti, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        # load pretrained model on miniImageNet
        pretrained_model = model_dict[params.model]()

        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        print("load from %s"%(modelfile))#"./logs/checkpoints/miniImagenet/ResNet10_baseline_aug/399.pth"

        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)#replace key name
            else:
                state.pop(key)#remove classifier
        pretrained_model.load_state_dict(state)#load checkpoints
        
        # train a new linear classifier
        classifier = Classifier(pretrained_model.final_feat_dim, n_way)#initializ only classifier with shape (512,5) for each task

        ###############################################################################################
        # split data into support set(5) and query set(15)
        n_query = x.size(1) - n_support
        #print(x.size())#torch.Size([5, 20, 3, 224, 224])
        #print(n_support)#5
        #print("n_query:%d"%(n_query))#15
        x = x.cuda()
        x_var = Variable(x)
        #print(x_var.data.shape)#torch.Size([5, 20, 3, 224, 224])
        # number of dataloaders is 5 and the real input is (20,3,224,224)
        #print(y)#however, y is useless and its shape is (5,20) => batch=5 and label=20

    
        batch_size = 4
        support_size = n_way * n_support#5*5=25  (maybe 5-way and each way contains 5 samples)
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda()
        #np.repeat(range( n_way ), n_support )=[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
        #print(y_a_i.data.shape)#torch.Size([25])


        #n_way=5 and n_query=15, view(75,3,224,224)
        #x_var[:, n_support:,:,:,:].shape=(5,15,3,224,224) => sample 5 loaders, where each contains a batch of images with shape (15,3,224,224)
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # query set
        #print(x_b_i.shape)#(75,3,224,224)  # 5 class loaders in total. Thus, batch size = 15*5 =75
        #x_b_i.shape=75,3,224,224
        #n_way * n_query ... (maybe 5-way and each way contains 15 samples)






        #n_way=5 and n_support=5, view(25,3,224,224)
        #x_var[:, :n_support,:,:,:].shape=(5,5,3,224,224)
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])  # support set
        #x_a_u.shape=25,3,224,224

        ################################################################################################
        # loss function and optimizer setting
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        

        if freeze_backbone is False:#for finetune use
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)


        pretrained_model.cuda()#pretrained on "mini-ImageNet" instead of "ImageNet"
        classifier.cuda()
        ###############################################################################################
        # fine-tuning
        #In the fine-tuning or meta-testing stage for all methods, we average the results over 600 experiments.
        #In each experiment, we randomly sample 5 classes from novel classes, and in each class, we also
        #pick k instances for the support set and 16 for the query set.
        #For Baseline and Baseline++, we use the entire support set to train a new classifier for 100 iterations with a batch size of 4.
        #For meta-learning methods, we obtain the classification model conditioned on the support set
        total_epoch = 100

        if freeze_backbone is False:#for finetune use
            pretrained_model.train()
        else:# if you don't want finetune
            pretrained_model.eval()
        
        classifier.train()#classifier should be dependent on task. Thus, we should update the classifier weights

        for epoch in range(total_epoch):#train classifier 100 epoch
            rand_id = np.random.permutation(support_size)#rand_id.shape=25
            #support_size=25
            #batch_size=4
            # using "support set" to train the classifier (and fine-tune the backbone).
            for j in range(0, support_size, batch_size):#support_size=25, batch_size=4
                classifier_opt.zero_grad()#clear classifier optimizer
                if freeze_backbone is False:#for finetune use
                    delta_opt.zero_grad()#update feature extractor

                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()#fetch only 4 elements

                #x_a_i.shape=25,3,224,224
                #y_a_i.shape=25
                z_batch = x_a_i[selected_id]#sample 4 inputs randomly from support set data
                #z_batch.shape=4,3,224,224

                #y_a_i=[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
                y_batch = y_a_i[selected_id]#sample 4 labels randomly from support set label
                #y_batch.shape=4

                output = pretrained_model(z_batch)#feature
                output = classifier(output)#predictions

                loss = loss_fn(output, y_batch)
                loss.backward()

                classifier_opt.step()#update classifier optimizer
                
                if freeze_backbone is False:#for finetune use
                    delta_opt.step()#update extractor

        ##############################################################################################
        # inference 
        pretrained_model.eval()
        classifier.eval()

        output = pretrained_model(x_b_i.cuda())#features
        scores = classifier(output)#predictions
       
        y_query = np.repeat(range( n_way ), n_query )#shape=(75)
        #y_query=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        #the 1st argument means return top-1
        #the 2nd argument dim=1 means return the value row-wisely
        #the 3rd arguemtn is largest=True
        #the 4th argument is sorted=True

        #topk_labels=[[1],[1], ..., [0],[0]] with shape (75,1)    cuz batch=75
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        acc_all.append((correct_this/ count_this *100))
        print("Task %d : %4.2f%%  Now avg: %4.2f%%" %(ti, correct_this/ count_this *100, np.mean(acc_all) ))
        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    task = params.task#fsl


    ##################################################################
    image_size = 224
    iter_num = 600
    #test_n_way=5
    #train_n_way=5
    #n_shot=5
    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
   
    
    ##################################################################
    # loading dataset 
    pretrained_dataset = "miniImageNet"
    
    novel_loaders = []
    if task == 'fsl':
        freeze_backbone = True

        dataset_names = ["miniImageNet"]
        print ("Loading mini-ImageNet")
        #iter_num=600, image_size=224, n_query=15, mode="test"
        datamgr             =  miniImageNet_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, mode="test", **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)#sample 5 images
        novel_loaders.append(novel_loader)
    else:
        freeze_backbone = params.freeze_backbone

        dataset_names = ["EuroSAT", "ISIC"]
        print ("Loading EuroSAT")
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
    
        print ("Loading ISIC")
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)

    #########################################################################
    # meta-test loop
    for idx, novel_loader in enumerate(novel_loaders):#[EuroSAT, ISIC]
        print("====")
        print (dataset_names[idx])#miniImageNet
        print("====")
        start_epoch = params.start_epoch#0
        stop_epoch = params.stop_epoch#400

        #pretrained_dataset=miniImageNet
        #novel_loader has 600 dataloaders
        #freeze_backbone=True
        #few_shot_params = dict(n_way = 5 , n_support = 5)
        meta_test(novel_loader,
            n_query = 15,
            pretrained_dataset=pretrained_dataset,
            freeze_backbone=freeze_backbone, **few_shot_params)
