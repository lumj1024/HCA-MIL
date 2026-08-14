#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:56:18 2021

@author: root
"""

from __future__ import print_function
import contextlib
import argparse
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
# torch.use_deterministic_algorithms(True)

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
# 数据集导入
from dataset import make_data
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib
matplotlib.use('agg')
import torch.backends.cudnn as cudnn
import random
import numpy as np
from tqdm import tqdm
#对比试验网络导入
# from model.MI_Netori import resnet18
# from model.Bimil import resnet18,permutation_consistency_loss
from model.Groupmamba.models.groupmambav9 import groupmamba_small, groupmamba_tiny,groupmamba_base
# from model.GDAMIL.GDAMIL import GDAMIL



# from model.vit5.vit5v1 import resnet18
# from model.vit5.models_vit5 import vit5_small
#消融实验网络导入
# from Ablation.NoMamba import resnet18
from Ablation.groupmambaablation import  get_ablation_model,ABLATION_CONFIGS


# from model.v9 import resnet18

from torch.optim import lr_scheduler
from loss import full_loss,calculate_objective

import matplotlib.pyplot as plt 
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batchSize', type=int, default=1, metavar='N',      #改一次送的图片张数
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N')    #整个数据集迭代多少次
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.002')    #学习速率,往小改            
parser.add_argument('--beta1', type=float, default=0.9,help='beta1 for adam. default=0.5')    #正则化
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 1)')
parser.add_argument('--epsilon', type=float, default=0.5)
parser.add_argument('--XI', type=float, default=1e-6)
parser.add_argument('--n_power', type=float, default=1)
parser.add_argument('--ablation', type=str, default='D_full',
                    choices=list(ABLATION_CONFIGS.keys()),
                    help='Ablation configuration name')

args = parser.parse_args()
print(args)   

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
set_random_seed(args.seed)          
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")
# === 对比实验:save_dir 自动按 --ablation 命名 ===
# save_path=["0","3407","GSRA_v9","GSRA_v8","GSRA_v7","Bimil","GSRA_v6","v5","Tinymamba","Ours","basemamba","tinymambaAtten","tinymambaMHLA","tinymambaDkmiltest",
#            "tinymambaDkmilLskAtten","vit5Atten"]

# === seq消融实验:save_dir 自动按 --ablation 命名 ===
# save_path=["seq24"]
# modelname = 0

# === 消融实验:save_dir 自动按 --ablation 命名 ===
save_dir_name = f'ablation_{args.ablation}'   # 例如 'ablation_A_baseline'

# 兼容原代码:保持 save_path 列表 + modelname 索引的结构
save_path = [save_dir_name]
modelname = 0


print("====>load traindatset ")
print('===> Loading datasets')
root_path = 'dataset/MILImage512s128/test/'
# seq_dir='/mnt/Disk1/lumingjie/data/leukemiadata/train/'
seq_dir='/dev/shm/lmjdata/leukemiadata/train/'
data_lists=glob(seq_dir+"*.npy")
# print(data_lists                  )

train_data, val_data = train_test_split(data_lists, test_size=0.2, random_state=42)
train_dataset = make_data(root_path,seq_dir,data_list=train_data,seq=16,extension=1,generate=0,stage="train")
      
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.threads,
                                           batch_size=args.batchSize, shuffle=True)
print('train_data_Num:'.format(len(train_loader)))

print("====>load valdatset ")
val_dataset = make_data(root_path,seq_dir,data_list=val_data,seq=16,extension=1,generate=0,stage="val")
      
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.threads,
                                           batch_size=args.batchSize, shuffle=True)

print('val_data_Num:'.format(len(val_loader)))

print("====>load testdatset ")
print('===> Loading datasets')
test_root_path = '/mnt/disk1/lumingjie/data/leukemiadata/test/'
# test_seq_dir='/mnt/Disk1/lumingjie/data/leukemiadata/test/'
test_seq_dir='/dev/shm/lmjdata/leukemiadata/test/'
test_dataset = make_data(test_root_path,test_seq_dir,seq=16,extension=1,generate=0,stage="test")
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=args.threads,
                                           batch_size=args.batchSize, shuffle=False)


print('data_Num:'.format(len(train_loader)))

def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(torch.abs(d.reshape(d.size(0), -1)), 1, keepdim=True)[0].view(d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    # d /= (1e-12 + torch.max(torch.abs(d.reshape(d.size(0), -1)), dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI,
                                              epsilon):
    d = torch.randn_like(x)

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit_m= model(x + d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, model, n_power, XI, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model,
                                                       n_power, XI, epsilon)
    logit_p = logit.detach()
    logit_m= model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
   
    return loss

class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model):
        super(VAT, self).__init__()
        self.model = model
        self.n_power =1
        self.XI = 1e-6
        self.epsilon= 0.5

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power,
                                            self.XI, self.epsilon)
#        print(vat_loss)
        return vat_loss  # already averaged
def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
  


print('===> Building model')
# model=resnet18(num_classes=2)
# ABLATION_NAME = args.ablation
# model = get_ablation_model(ABLATION_NAME, num_classes=2)
model=groupmamba_tiny(num_classes=2)
print(model)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
reg_fn = VAT(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-06)

print('===> Begining Training')

def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0.0
    train_loss = 0.0
    total =0
    train_iterator = tqdm(train_loader, leave=True, total=len(train_loader), position=0)
    iterators=0
    margins_loss_epoch = torch.tensor([0.], requires_grad=True).to(device)
    
    for  _,inputs, labels in train_iterator:
        # torch.cuda.empty_cache()
        iterators=iterators+1
        images=inputs.to(device)
        labels=labels.to(device)
        images = images.squeeze(0)
        images=images.permute(0, 3, 1, 2)
       
        optimizer.zero_grad()
      
        outputs = model(images)
        #稳定
        loss_vat = reg_fn(images, outputs)
        loss_ce = criterion(outputs, labels)
        loss_el= entropy_loss(outputs)
        loss =  loss_ce+ loss_el+loss_vat

        # loss_pc  = permutation_consistency_loss(model, images)   # ★ 新增
        # loss = loss_ce + loss_el + loss_vat + 0.1 * loss_pc      # ★ 权重 0.1,后续可调

        # loss = criterion(outputs, labels)
        train_loss += loss.item()
        total += labels.size(0)
#
        _, prediction = torch.max(outputs.data, 1)

        train_acc += torch.sum(prediction == labels)
        status="===> Epoch[{}]({}/{}): train_loss:{:.4f},mean_loss:{:.4f}, train_acc:{:.4f}".format(
        epoch, iterators, len(train_loader), loss.item(),train_loss/total,train_acc/total)
        # print(status)
        train_iterator.set_description(status)
        loss.backward()
        optimizer.step()
        
#    scheduler.step(loss)
    return train_acc/total,train_loss/total

def val(epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        val_acc = 0.0
        val_acc0 = 0.0
        val_acc1 = 0.0
        
        val_loss=0
       
        total =0
        total0 =0
        total1 =0
        val_iterator = tqdm(val_loader, leave=True, total=len(val_loader), position=0)
        iterators=0
        for iteration, (_,images, labels)  in enumerate(val_iterator):
            iterators=iterators+1
            images=images.to(device)
           
            labels=labels.to(device)
            images = images.squeeze(0)
            images=images.permute(0, 3, 1, 2)
        
            outputs = model(images)
            outputs=F.softmax(outputs)
            v_loss = criterion(outputs, labels)
        
     
            val_loss += v_loss.item()
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            val_acc += torch.sum(prediction == labels)
            
           
            index0=(labels == 0).nonzero()
            total0 += index0.size(0)
            val_acc0 += torch.sum(prediction[index0] == labels[index0])
                
            
            index1=(labels == 1).nonzero()
            total1 += index1.size(0)
            val_acc1 += torch.sum(prediction[index1] == labels[index1])
            
            
           
    #        
           
            # print("===> Epoch[{}] =====>Mean_val_Acc:{:.4f},ALL_Acc:{:.4f},CLL_Acc:{:.4f},".format(
            #     epoch, val_acc/total,val_acc0/total0,val_acc1/total1))
            # print(" =====pred ALL:{}".format(np.array(prediction[index0].cpu().detach().numpy()).T))
            # print(" =====pred CLL:{}".format(np.array(prediction[index1].cpu().detach().numpy()).T))
            
    return val_acc/total,val_loss/total

def test(epoch):
    with torch.no_grad():
        # torch.cuda.empty_cache()
        model.eval()
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
    
       
        total =0
        total0 =0
        total1 =0
        
        test_iterator = tqdm(test_loader, leave=True, total=len(test_loader), position=0)
        iterators = 0
        for iteration, (_,images, labels)  in enumerate(test_iterator):
            iterators=iterators+1
            images=images.to(device)
           
            labels=labels.to(device)
            images = images.squeeze(0)
            images=images.permute(0, 3, 1, 2)
            outputs = model(images)
            outputs=F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
            
           
            index0=(labels == 0).nonzero()
            total0 += index0.size(0)
            test_acc0 += torch.sum(prediction[index0] == labels[index0])
                
            
            index1=(labels == 1).nonzero()
            total1 += index1.size(0)
            test_acc1 += torch.sum(prediction[index1] == labels[index1])

         
          
        # print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f},ALL_Acc:{:.4f},CLL_Acc:{:.4f}".format(
        #         epoch, test_acc/total,test_acc0/total0,test_acc1/total1))
        print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f},Acc0:{:.4f},Acc1:{:.4f}".format(
        epoch, test_acc / total, test_acc0 / total0, test_acc1 / total1))
        write_to_file(epoch, test_acc / total, test_acc0 / total0, test_acc1 / total1)
        # print(" =====pred ET:{}".format(np.array(prediction[index0].cpu().detach().numpy()).T))
        # print(" =====pred overt-MF:{}".format(np.array(prediction[index1].cpu().detach().numpy()).T))
        # print(" =====pred pre-MF:{}".format(np.array(prediction[index2].cpu().detach().numpy()).T))
        # print(" =====pred PV:{}".format(np.array(prediction[index3].cpu().detach().numpy()).T))
    return test_acc/total
def write_to_file(epoch, acc ,acc0, acc1, file_path='./log/{}/result.txt'.format(save_path[modelname])):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: {epoch}, Acc: {acc} Acc0: {acc0}, Acc1: {acc1}\n")

def checkpoint(name):
    model_out_path = name
    torch.save(model.state_dict(), model_out_path)
    print("\n===>Checkpoint saved to {}".format(model_out_path))
        
def show_curve(total_loss_curve, plot_title='total_loss', show = False, save = False, path = 'Train_curve.png'):
    x = range(1,len(total_loss_curve)+1)
    plt.plot(x, total_loss_curve)
    plt.title(plot_title) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
#plot loss
def show_loss(total_loss_curve, plot_title='total_loss', show = False, save = False, path = 'Train_curve.png'):
    x = range(1,len(total_loss_curve)+1)
    plt.plot(x, total_loss_curve)
    plt.title(plot_title) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()  

def show_curve_two(test_acc_curve, val_acc_curve,plot_title='total_loss', show = False, save = False, path = 'Train_curve.png'):
    x = range(1,len(test_acc_curve)+1)
    plt.plot(x,test_acc_curve,label='test')
    plt.plot(x, val_acc_curve,label='train')
    plt.title(plot_title) 
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()        

def show_curve_there(train_acc_curve,test_acc_curve, val_acc_curve,plot_title='total_loss', show = False, save = False, path = 'Train_curve.png'):
    x = range(1,len(test_acc_curve)+1)
    plt.plot(x,train_acc_curve,label='train')
    plt.plot(x,test_acc_curve,label='test')
    plt.plot(x, val_acc_curve,label='val')
    plt.title(plot_title) 
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()   
           
if __name__ == '__main__' :
        import os
        val_best_acc=0
        test_best_acc=0
        val_best_loss=10
        min_epoch=10
        if not os.path.exists('./log/{}'.format(save_path[modelname])):
            os.makedirs('./log/{}'.format(save_path[modelname]))
        if not os.path.exists('./log/{}/checkpoint'.format(save_path[modelname])):
            os.makedirs('./log/{}/checkpoint'.format(save_path[modelname]))
        if not os.path.exists('./log/{}/savefile'.format(save_path[modelname])):
            os.makedirs('./log/{}/savefile'.format(save_path[modelname]))
            
        log_dir='./log/{}/checkpoint/model_min.pth'.format(save_path[modelname])
        last_log='./log/{}/checkpoint/min_last.pth'.format(save_path[modelname])
        best_log='./log/{}/checkpoint/min_best.pth'.format(save_path[modelname])
        total_ave_loss_list=[]
        train_loss_list=[]
        test_loss_list=[]
        val_loss_list=[]

        test_acc_list=[]
        train_acc_list=[]
        val_acc_list=[]
       
        if os.path.exists(log_dir):

            model.load_state_dict(torch.load(log_dir))
            print('load_weight')
            if 0:
                for param in model.parameters(): 
                    param.requires_grad = False
                print(model.fc)
#          RTUY
                
            model.to(device)  
            print('finetuing model')
        count=0
        for epoch in range(1, args.epochs + 1): 
            log_dir='./log/{}/checkpoint/model_min_{}.pth'.format(save_path[modelname],epoch)
            train_acc,train_loss=train(epoch)
            test_acc=test(epoch)
            val_acc,val_loss=val(epoch)

            # total_ave_loss_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            test_acc_list.append(test_acc.cpu().data.numpy())
            train_acc_list.append(train_acc.cpu().data.numpy())
            val_acc_list.append(val_acc.cpu().data.numpy())

            
            show_loss(train_loss_list, plot_title='train_loss', show = False, save =True, path = './log/{}/train_loss.png'.format(save_path[modelname]))
            show_loss(val_loss_list, plot_title='val_loss', show = False, save =True, path = './log/{}/val_loss.png'.format(save_path[modelname]))
            
            show_curve(train_acc_list, plot_title='train_acc', show = False, save =True, path = './log/{}/train_acc.png'.format(save_path[modelname]))
            show_curve(val_acc_list, plot_title='val_acc', show = False, save =True, path = './log/{}/val_acc.png'.format(save_path[modelname]))
            # show_curve_two(test_acc_list, val_acc_list,plot_title='two_acc', show = False, save =True, path = '/mnt/Disk1/lianchentao/MIL_RNTtest/log/{}/test_acc.png'.format(save_path))
            show_curve_there(train_acc_list,test_acc_list, val_acc_list,plot_title='there_acc', show = False, save =True, path = './log/{}/test_acc.png'.format(save_path[modelname]))
            
            model_save_dir = './log/{}/checkpoint'.format(save_path[modelname])
            os.makedirs(model_save_dir, exist_ok=True)
            epoch_model_path = os.path.join(model_save_dir, f'epoch_{epoch}.pth')
            # checkpoint(epoch_model_path)
           
            if epoch==1:
                 val_best_loss=  val_loss
                 test_best_acc= test_acc
                 val_best_acc=  val_acc
            
            if val_acc>val_best_acc:
            #    best = epoch
            #    log_dir='./log/{}/checkpoint/model_min_{}.pth'.format(save_path[modelname],best)
               checkpoint(log_dir)
               val_best_loss = val_loss
               test_best_acc = test_acc
               val_best_acc = val_acc
               count = 0
             
            else:
                val_best_loss = val_best_loss
                test_best_acc = test_best_acc
                val_best_acc = val_best_acc
                count=count+1
                print(count)
                
            checkpoint(last_log)
            if count>50:
                checkpoint(best_log)
                np.save('./log/{}/savefile/train_data.npy'.format(save_path[modelname]), train_loss_list)
                np.save('./log/{}/savefile/val_data.npy'.format(save_path[modelname]), val_loss_list)
                np.save('./log/{}/savefile/train_acc_data.npy'.format(save_path[modelname]), train_acc_list)
                np.save('./log/{}/savefile/test_acc_data.npy'.format(save_path[modelname]), test_acc_list)
                np.save('./log/{}/savefile/val_acc_data.npy'.format(save_path[modelname]), val_acc_list)
                break
#            scheduler.step()
            print("\n===>val_acc not be improved  to {:.4f}".format( val_best_acc))
            print("\n===>test_acc not be improved  to {:.4f}".format( test_best_acc))