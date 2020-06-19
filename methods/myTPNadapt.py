# This code is modified from https://github.com/csyanbin/TPN-pytorch
# This code is modified from https://github.com/jvanvugt/pytorch-domain-adaptation

import backbone
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import pdb
import utils
import numpy as np
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.domain_clf = nn.Sequential(
            GradientReversal(),
            nn.Linear(3*3, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

    def forward(self, x):
        x = x.view(-1,512,7,7)
        out = self.layer1(x)#torch.Size([105, 64, 4, 4])
        out = self.layer2(out)#torch.Size([105, 1, 3, 3])
        #flatten
        out = out.view(out.size(0),-1)#(105,9)
        out = self.domain_clf(out)#(105,1)
        return out.squeeze()#(105)
#discriminator = DomainClassifier()
#domain_preds = discriminator(features)
#domin_preds.shape=torch.Size([64]) float32
#domain_y.shape=torch.Size([64])  float32
#domain_y=[32 times 1 ... 32 times 0]
#domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
#loss = domain_loss + label_loss



class RelationNetwork(nn.Module):
    #Graph Construction Module
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.fc3 = nn.Linear(3*3, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.view(-1,512,7,7)
        #pdb.set_trace()
        out = self.layer1(x)#torch.Size([105, 64, 4, 4])
        out = self.layer2(out)#torch.Size([105, 1, 3, 3])
        # flatten
        out = out.view(out.size(0),-1)
        #print("out.shape in RelationNetwork:",out.shape)#(105,9)
        out = F.relu(self.fc3(out))#(105,8)
        #print(out.shape)
        out = self.fc4(out) # (105,1)
        #print(out.shape)#(105,1)
        out = out.view(out.size(0),-1) # bs*1
        #print(out.shape)#(105,1)
        #print("============")
        return out

class MyTPN_Adapt(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(MyTPN_Adapt, self).__init__( lambda: None,  n_way, n_support)
        self.feature = model_func(flatten=False).cuda()
        self.loss_fn  = nn.CrossEntropyLoss().cuda()
        self.relation = RelationNetwork().cuda()
        self.domainclf = DomainClassifier().cuda()
        self.it = 0
        self.alpha = torch.tensor([0.99], requires_grad=False).cuda()
    
    def set_forward(self, x, is_feature=False):
        s_labels, q_labels, F, Fq = self.set_forward_all(x)
        return Fq


    def set_forward_all(self,x):#remember to set is_feature arguments otherwise some weired bugs happened
        eps = np.finfo(float).eps
        #pdb.set_trace()
        #pdb.set_trace()
        #x.shape=(5,20,3,224,224)
        s_labels = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support))#[25]
        q_labels = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query  ))#[75]
        s_labels = s_labels.type(torch.LongTensor)
        q_labels = q_labels.type(torch.LongTensor)

        s_labels = torch.zeros(self.n_way * self.n_support, self.n_way).scatter_(1, s_labels.view(-1,1), 1)#torch.Size([25, 5])
        q_labels = torch.zeros(self.n_way * self.n_query, self.n_way).scatter_(1, q_labels.view(-1,1), 1)#torch.Size([75, 5])
        s_labels = s_labels.cuda()
        q_labels = q_labels.cuda()



        
        #x.shape=(5,20,3,84,84)
        #x    = Variable(x.to(self.device))
        #inputs = [x.cuda(0), s_onehot.cuda(0), q_onehot.cuda(0)]
        x = x.cuda()
        #inp  = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])#100,3,84,84
        #pdb.set_trace()
        query = x[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:])
        support = x[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])
        inp     = torch.cat((support,query), 0)
        
        emb_all = self.feature(inp)
        #print("emb_all.shape:",emb_all.shape)#(105,512,7,7)    => (105,25088)
        
        emb_all = emb_all.view(-1,25088)#torch.Size([100, 1600]) #4-order to 2-order
        #25088=512*7*7
        #emb_all = emb_all.view(-1,1600)
        N, d    = emb_all.shape[0], emb_all.shape[1]#100,1600


        
        self.sigma = self.relation(emb_all)#(100,1600)
        #pdb.set_trace()
        
        emb_all = emb_all / (self.sigma+eps) # N*d
        emb1    = torch.unsqueeze(emb_all,1) # N*1*d
        emb2    = torch.unsqueeze(emb_all,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/2)




        _, indices = torch.topk(W, 20)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)#create binary mask
        mask = ((mask+torch.t(mask))>0).type(torch.float32)# union, kNN graph
        W    = W*mask
        ## normalize
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2
        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        #pdb.set_trace()
        ys = s_labels#torch.Size(25,5) torch.float32
        yu = torch.zeros(self.n_way * self.n_query, self.n_way).cuda()#(75,5)
        #ys = ys.cuda()
        #yu = yu.cuda()
        y  = torch.cat((ys,yu),0)#(100,5)
        #pdb.set_trace()
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda()-self.alpha*S+eps), y)
        Fq = F[self.n_way*self.n_support:, :] # query predictions (75,5)
        return s_labels, q_labels, F, Fq
    



    def set_forward_metatrain_all(self,x):#remember to set is_feature arguments otherwise some weired bugs happened
        eps = np.finfo(float).eps
        #pdb.set_trace()
        #pdb.set_trace()
        #x.shape=(5,20,3,224,224)
        s_labels = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support))#[25]
        q_labels = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query  ))#[75]
        s_labels = s_labels.type(torch.LongTensor)
        q_labels = q_labels.type(torch.LongTensor)

        s_labels = torch.zeros(self.n_way * self.n_support, self.n_way).scatter_(1, s_labels.view(-1,1), 1)#torch.Size([25, 5])
        q_labels = torch.zeros(self.n_way * self.n_query, self.n_way).scatter_(1, q_labels.view(-1,1), 1)#torch.Size([75, 5])
        s_labels = s_labels.cuda()
        q_labels = q_labels.cuda()



        
        #x.shape=(5,20,3,84,84)
        #x    = Variable(x.to(self.device))
        #inputs = [x.cuda(0), s_onehot.cuda(0), q_onehot.cuda(0)]
        x = x.cuda()
        #pdb.set_trace()#x.shape=(10,21,3,224,224)
        #inp  = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])#100,3,84,84
        #pdb.set_trace()
        query = x[:x.shape[0]//2, self.n_support:,:,:,:].contiguous().view( self.n_way * self.n_query,   *x.size()[2:])
        #query.shape=(80,3,224,224)
        support = x[:x.shape[0]//2,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])
        #support.shape=(25,3,224,224)
        novel = x[x.shape[0]//2:,:,:,:,:].contiguous().view( self.n_way*(self.n_support+self.n_query), *x.size()[2:])#new-added
        #novel.shape=(105,3,224,224)

        inp     = torch.cat((support,query,novel), 0)#torch.Size([210,3,224,224])


        
        emb_all = self.feature(inp)#emb_all.shape=(210,512,7,7)
        
        emb_all = emb_all.view(-1,25088)#torch.Size([210, 25088]) #4-order to 2-order
        #25088=512*7*7
        #emb_all = emb_all.view(-1,1600)
        N, d    = emb_all.shape[0]//2, emb_all.shape[1]#105,25088
        
        self.sigma = self.relation(emb_all[:emb_all.shape[0]//2])#(105,25088)


        #================= domain adaptation ===========
        domain_preds = self.domainclf(emb_all)
        #===============================================
        #pdb.set_trace()
        
        part_emb_all = emb_all[:emb_all.shape[0]//2] / (self.sigma+eps) # N*d
        emb1    = torch.unsqueeze(part_emb_all,1) # N*1*d
        emb2    = torch.unsqueeze(part_emb_all,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/2)




        _, indices = torch.topk(W, 20)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)#create binary mask
        mask = ((mask+torch.t(mask))>0).type(torch.float32)# union, kNN graph
        W    = W*mask
        ## normalize
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2
        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        #pdb.set_trace()
        ys = s_labels#torch.Size(25,5) torch.float32
        yu = torch.zeros(self.n_way * self.n_query, self.n_way).cuda()#(75,5)
        y  = torch.cat((ys,yu),0)#(100,5)
        #pdb.set_trace()
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda()-self.alpha*S+eps), y)
        Fq = F[self.n_way*self.n_support:, :] # query predictions (75,5)
        return s_labels, q_labels, F, Fq, domain_preds


    def set_forward_loss(self, x):
        s_labels, q_labels, FF, Fq, domain_preds = self.set_forward_metatrain_all(x)
        answer = torch.cat((s_labels, q_labels), 0)#(105,5)
        gt = torch.argmax(answer, 1)#torch.Size([105])   torch.int64
        #gt = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
        #0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        #3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])

        #pdb.set_trace()#domain_preds.shape=torch.size([210])
        domain_y = torch.cat([torch.ones(domain_preds.shape[0]//2),torch.zeros(domain_preds.shape[0]//2)],0).cuda()
        #gt = gt.cuda()
        loss = self.loss_fn(FF, gt)

        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
        if self.it % 100 == 0:
            print("label loss:", loss.item())
            print("domain loss:",domain_loss.item())
        self.it+=1

        loss += domain_loss
        return loss

    def correct(self, x):
        #self.it+=1
        #pdb.set_trace()
        s_labels, q_labels, F, Fq = self.set_forward_all(x)




        Fs = F[:self.n_way*self.n_support, :]
        preds = torch.argmax(Fs,1)
        gts   = torch.argmax(s_labels,1)
        top1_correct = (preds==gts).sum()
        """
        if self.it % 300 == 0:
            #print(torch.argmax(F,1))
            print("[SUPPORT SET] predictions: ",preds)
            print("train acc:",float(top1_correct) / len(s_labels))
        """
        
        #answer = torch.cat((s_labels, q_labels), 0)
        #gt = torch.argmax(answer, 1)





        # compute clssification accuracy
        predq = torch.argmax(Fq,1)#find argmax in query predictions
        gtq   = torch.argmax(q_labels,1)#find argmax in grouth truth
        #gtq = gtq.cuda()
        top1_correct = (predq==gtq).sum()
        y_query = q_labels
        #pdb.set_trace()
        """
        if self.it % 300 == 0:
            print("predictions: ",predq)
            print("current accu: %f"%(float(top1_correct) / len(y_query)))
        """
        return float(top1_correct), len(y_query)
    
    def train_loop(self, epoch, base_loader, novel_loader, optimizer ):
        #print("overwrite train_loop")
        print_freq = 10
        avg_loss=0
        batches = zip(base_loader, novel_loader)#plz check this line
        for i, ((base_x, _), (novel_x, _)) in enumerate(batches):
            #pdb.set_trace()
            #base_x.shape=(5,20,3,224,224)
            self.n_query = base_x.size(1) - self.n_support#16
            if self.change_way:#True
                self.n_way = base_x.size(0)#5
            #base_x.shape=torch.Size([5, 21, 3, 224, 224])
            #novel_x.shape=torch.Size([5, 21, 3, 224, 224])
            x = torch.cat([base_x, novel_x], 0)#x.shape=torch.Size([10, 21, 3, 224, 224])
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()
            #if i == 50:break
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(base_loader), avg_loss/float(i+1)))
    
    def load_my_state_dict(self, state_dict):
        #https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state: continue
            if isinstance(param, Parameter): param = param.data# backwards compatibility for serialized parameters
            own_state[name].copy_(param)