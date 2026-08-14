#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified train.py for baseline comparison experiments.

USAGE:
  Just change MODEL_NAME (line ~60) to one of:
    'v9'              -> your full method (GroupMamba + your modules), 96%
    'resnet18'        -> CNN baseline
    'resnet50'        -> CNN baseline
    'convnext_tiny'   -> modern CNN
    'vit_small'       -> Transformer baseline
    'swin_tiny'       -> Hierarchical Transformer
    'groupmamba_tiny' -> pure GroupMamba (no your modules) — for ablation

Everything else (VAT, entropy loss, scheduler, train loop, logging) stays
identical so that all baselines are compared under the SAME training
protocol.

Logs go to ./log/baseline_<MODEL_NAME>/  to avoid overwriting your v9 results.
"""

from __future__ import print_function
import contextlib
import argparse
import os
import torch

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

# ============================================================
# 模型导入 — 你的完整方法 (v9) + baseline backbone wrapper
# ============================================================
from model.Groupmamba.models.groupmambav9 import groupmamba_tiny as v9_model
from model.backbones import build_model


from torch.optim import lr_scheduler
from loss import full_loss, calculate_objective

import matplotlib.pyplot as plt

# ============================================================
# ★★★ 切换实验只需要改这一行 ★★★
# ============================================================
# Options:
#   'v9'              -> your full method (groupmambav9), expected ~96%
#   'resnet18'        -> CNN baseline
#   'resnet50'        -> CNN baseline (most common)
#   'convnext_tiny'   -> modern CNN
#   'vit_small'       -> Transformer
#   'swin_tiny'       -> Hierarchical Transformer
#   'groupmamba_tiny' -> pure GroupMamba (no your modules) — KEY ablation
MODEL_NAME = 'groupmamba_tiny'
# ============================================================
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batchSize', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
parser.add_argument('--epsilon', type=float, default=0.5)
parser.add_argument('--XI', type=float, default=1e-6)
parser.add_argument('--n_power', type=float, default=1)
# Allow overriding MODEL_NAME from command line:
# python train.py --model resnet50
parser.add_argument('--model', type=str, default=None,
                    help='Override MODEL_NAME from command line.')

args = parser.parse_args()
print(args)

# Override MODEL_NAME if provided via --model
if args.model is not None:
    MODEL_NAME = args.model
print(f"\n[experiment] MODEL_NAME = '{MODEL_NAME}'\n")


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(args.seed)


# ============================================================
# Save path: separate dir per experiment to avoid overwriting
# ============================================================
if MODEL_NAME == 'v9':
    save_dir_name = 'GSRA_v9'   # 跟你之前的命名一致
else:
    save_dir_name = f'baseline_{MODEL_NAME}'


print("====>load traindatset ")
print('===> Loading datasets')
root_path = 'dataset/MILImage512s128/test/'
seq_dir = '/dev/shm/lmjdata/leukemiadata/train/'
data_lists = glob(seq_dir + "*.npy")

train_data, val_data = train_test_split(data_lists, test_size=0.2, random_state=42)
train_dataset = make_data(root_path, seq_dir, data_list=train_data, seq=16, extension=1, generate=0, stage="train")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.threads,
                                           batch_size=args.batchSize, shuffle=True)
print('train_data_Num:'.format(len(train_loader)))

print("====>load valdatset ")
val_dataset = make_data(root_path, seq_dir, data_list=val_data, seq=16, extension=1, generate=0, stage="val")

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.threads,
                                         batch_size=args.batchSize, shuffle=True)

print('val_data_Num:'.format(len(val_loader)))

print("====>load testdatset ")
print('===> Loading datasets')
test_root_path = '/mnt/disk1/lumingjie/data/leukemiadata/test/'
test_seq_dir = '/dev/shm/lmjdata/leukemiadata/test/'
test_dataset = make_data(test_root_path, test_seq_dir, seq=16, extension=1, generate=0, stage="test")
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
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI, epsilon):
    d = torch.randn_like(x)
    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()
    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, model, n_power, XI, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI, epsilon)
    logit_p = logit.detach()
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return loss


class VAT(nn.Module):
    def __init__(self, model):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = 1
        self.XI = 1e-6
        self.epsilon = 0.5

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power,
                                            self.XI, self.epsilon)
        return vat_loss


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


# ============================================================
# Build model based on MODEL_NAME
# ============================================================
print('===> Building model')

if MODEL_NAME == 'v9':
    # Your full method
    model = v9_model(num_classes=2)
else:
    # Baseline backbone (CNN / Transformer / pure GroupMamba) with ABMIL pooling
    # All baselines share the same input size (256x256) as your dataset.
    model = build_model(MODEL_NAME, num_classes=2, img_size=256)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[model] '{MODEL_NAME}' built. Trainable params: {n_params/1e6:.2f} M")

criterion = nn.CrossEntropyLoss()
model = model.to(device)
reg_fn = VAT(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-06)

print('===> Begining Training')


def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0.0
    train_loss = 0.0
    total = 0
    train_iterator = tqdm(train_loader, leave=True, total=len(train_loader), position=0)
    iterators = 0

    for _, inputs, labels in train_iterator:
        iterators = iterators + 1
        images = inputs.to(device)
        labels = labels.to(device)
        images = images.squeeze(0)
        images = images.permute(0, 3, 1, 2)

        optimizer.zero_grad()

        outputs = model(images)
        loss_vat = reg_fn(images, outputs)
        loss_ce = criterion(outputs, labels)
        loss_el = entropy_loss(outputs)
        loss = loss_ce + loss_el + loss_vat

        train_loss += loss.item()
        total += labels.size(0)

        _, prediction = torch.max(outputs.data, 1)
        train_acc += torch.sum(prediction == labels)
        status = "===> Epoch[{}]({}/{}): train_loss:{:.4f},mean_loss:{:.4f}, train_acc:{:.4f}".format(
            epoch, iterators, len(train_loader), loss.item(), train_loss / total, train_acc / total)
        train_iterator.set_description(status)
        loss.backward()
        optimizer.step()

    return train_acc / total, train_loss / total


def val(epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        val_acc = 0.0
        val_acc0 = 0.0
        val_acc1 = 0.0
        val_loss = 0
        total = 0
        total0 = 0
        total1 = 0
        val_iterator = tqdm(val_loader, leave=True, total=len(val_loader), position=0)
        iterators = 0
        for iteration, (_, images, labels) in enumerate(val_iterator):
            iterators = iterators + 1
            images = images.to(device)
            labels = labels.to(device)
            images = images.squeeze(0)
            images = images.permute(0, 3, 1, 2)

            outputs = model(images)
            outputs = F.softmax(outputs)
            v_loss = criterion(outputs, labels)

            val_loss += v_loss.item()
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            val_acc += torch.sum(prediction == labels)

            index0 = (labels == 0).nonzero()
            total0 += index0.size(0)
            val_acc0 += torch.sum(prediction[index0] == labels[index0])

            index1 = (labels == 1).nonzero()
            total1 += index1.size(0)
            val_acc1 += torch.sum(prediction[index1] == labels[index1])

    return val_acc / total, val_loss / total


def test(epoch):
    with torch.no_grad():
        model.eval()
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
        total = 0
        total0 = 0
        total1 = 0

        test_iterator = tqdm(test_loader, leave=True, total=len(test_loader), position=0)
        iterators = 0
        for iteration, (_, images, labels) in enumerate(test_iterator):
            iterators = iterators + 1
            images = images.to(device)
            labels = labels.to(device)
            images = images.squeeze(0)
            images = images.permute(0, 3, 1, 2)
            outputs = model(images)
            outputs = F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)

            index0 = (labels == 0).nonzero()
            total0 += index0.size(0)
            test_acc0 += torch.sum(prediction[index0] == labels[index0])

            index1 = (labels == 1).nonzero()
            total1 += index1.size(0)
            test_acc1 += torch.sum(prediction[index1] == labels[index1])

        # Avoid divide-by-zero on first epoch when one class has no samples in a split
        acc0 = test_acc0 / total0 if total0 > 0 else torch.tensor(0.0)
        acc1 = test_acc1 / total1 if total1 > 0 else torch.tensor(0.0)

        print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f},Acc0:{:.4f},Acc1:{:.4f}".format(
            epoch, test_acc / total, acc0, acc1))
        write_to_file(epoch, test_acc / total, acc0, acc1)
    return test_acc / total


def write_to_file(epoch, acc, acc0, acc1, file_path=None):
    if file_path is None:
        file_path = './log/{}/result.txt'.format(save_dir_name)
    with open(file_path, 'a') as file:
        file.write(f"Epoch: {epoch}, Acc: {acc} Acc0: {acc0}, Acc1: {acc1}\n")


def checkpoint(name):
    model_out_path = name
    torch.save(model.state_dict(), model_out_path)
    print("\n===>Checkpoint saved to {}".format(model_out_path))


def show_curve(total_loss_curve, plot_title='total_loss', show=False, save=False, path='Train_curve.png'):
    x = range(1, len(total_loss_curve) + 1)
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


def show_loss(total_loss_curve, plot_title='total_loss', show=False, save=False, path='Train_curve.png'):
    x = range(1, len(total_loss_curve) + 1)
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


def show_curve_two(test_acc_curve, val_acc_curve, plot_title='total_loss', show=False, save=False, path='Train_curve.png'):
    x = range(1, len(test_acc_curve) + 1)
    plt.plot(x, test_acc_curve, label='test')
    plt.plot(x, val_acc_curve, label='train')
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


def show_curve_there(train_acc_curve, test_acc_curve, val_acc_curve, plot_title='total_loss', show=False, save=False, path='Train_curve.png'):
    x = range(1, len(test_acc_curve) + 1)
    plt.plot(x, train_acc_curve, label='train')
    plt.plot(x, test_acc_curve, label='test')
    plt.plot(x, val_acc_curve, label='val')
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


if __name__ == '__main__':
    val_best_acc = 0
    test_best_acc = 0
    val_best_loss = 10
    min_epoch = 10

    if not os.path.exists('./log/{}'.format(save_dir_name)):
        os.makedirs('./log/{}'.format(save_dir_name))
    if not os.path.exists('./log/{}/checkpoint'.format(save_dir_name)):
        os.makedirs('./log/{}/checkpoint'.format(save_dir_name))
    if not os.path.exists('./log/{}/savefile'.format(save_dir_name)):
        os.makedirs('./log/{}/savefile'.format(save_dir_name))

    log_dir = './log/{}/checkpoint/model_min.pth'.format(save_dir_name)
    last_log = './log/{}/checkpoint/min_last.pth'.format(save_dir_name)
    best_log = './log/{}/checkpoint/min_best.pth'.format(save_dir_name)

    train_loss_list = []
    test_loss_list = []
    val_loss_list = []
    test_acc_list = []
    train_acc_list = []
    val_acc_list = []

    if os.path.exists(log_dir):
        try:
            model.load_state_dict(torch.load(log_dir))
            print('load_weight')
            model.to(device)
            print('finetuing model')
        except Exception as e:
            print(f'[warn] failed to load existing checkpoint: {e}')
            print('[warn] starting from scratch')

    count = 0
    for epoch in range(1, args.epochs + 1):
        log_dir = './log/{}/checkpoint/model_min_{}.pth'.format(save_dir_name, epoch)
        train_acc, train_loss = train(epoch)
        test_acc = test(epoch)
        val_acc, val_loss = val(epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_acc_list.append(test_acc.cpu().data.numpy())
        train_acc_list.append(train_acc.cpu().data.numpy())
        val_acc_list.append(val_acc.cpu().data.numpy())

        show_loss(train_loss_list, plot_title='train_loss', show=False, save=True,
                  path='./log/{}/train_loss.png'.format(save_dir_name))
        show_loss(val_loss_list, plot_title='val_loss', show=False, save=True,
                  path='./log/{}/val_loss.png'.format(save_dir_name))
        show_curve(train_acc_list, plot_title='train_acc', show=False, save=True,
                   path='./log/{}/train_acc.png'.format(save_dir_name))
        show_curve(val_acc_list, plot_title='val_acc', show=False, save=True,
                   path='./log/{}/val_acc.png'.format(save_dir_name))
        show_curve_there(train_acc_list, test_acc_list, val_acc_list, plot_title='there_acc',
                         show=False, save=True,
                         path='./log/{}/test_acc.png'.format(save_dir_name))

        model_save_dir = './log/{}/checkpoint'.format(save_dir_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if epoch == 1:
            val_best_loss = val_loss
            test_best_acc = test_acc
            val_best_acc = val_acc

        if val_acc > val_best_acc:
            checkpoint(log_dir)
            val_best_loss = val_loss
            test_best_acc = test_acc
            val_best_acc = val_acc
            count = 0
        else:
            count = count + 1
            print(count)

        checkpoint(last_log)
        if count > 50:
            checkpoint(best_log)
            np.save('./log/{}/savefile/train_data.npy'.format(save_dir_name), train_loss_list)
            np.save('./log/{}/savefile/val_data.npy'.format(save_dir_name), val_loss_list)
            np.save('./log/{}/savefile/train_acc_data.npy'.format(save_dir_name), train_acc_list)
            np.save('./log/{}/savefile/test_acc_data.npy'.format(save_dir_name), test_acc_list)
            np.save('./log/{}/savefile/val_acc_data.npy'.format(save_dir_name), val_acc_list)
            break

        print("\n===>val_acc not be improved  to {:.4f}".format(val_best_acc))
        print("\n===>test_acc not be improved  to {:.4f}".format(test_best_acc))