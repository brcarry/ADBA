#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
import math
import pdb
import torch
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from helpers.datasets import partition_data
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, kldiv, add_trigger, test_trigger_accuracy
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
from torch.utils.data import DataLoader, Dataset
from PIL import Image



warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)



class LocalUpdate_with_mask(object):
    def __init__(self, args, dataset):
        self.args = args
        self.train_loader = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        
    def update_weights(self, model):
        init_mask = np.zeros((1, 32, 32)).astype(np.float32)
        init_pattern = np.random.normal(0, 1, (3, 32, 32)).astype(np.float32)
        mask_nc = torch.from_numpy(init_mask).clamp_(0, 1) # nc means neural cleanse
        pattern_nc = torch.from_numpy(init_pattern).clamp_(0, 1)
        
        shadow_model = resnet18(num_classes=10).cuda()
        
        pattern = pattern_nc.cuda()
        mask = mask_nc.cuda()
        pattern.requires_grad_(True)
        mask.requires_grad_(True)

        target_label = self.args.target_label_backdoor

        optimizer_for_shadow = torch.optim.SGD(shadow_model.parameters(), lr=self.args.lr, momentum=0.9)
        optimizer_for_teacher = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        optimizer_for_trigger = torch.optim.SGD([pattern, mask], lr=self.args.lr, momentum=0.9)

        local_acc_list = []
        
        print("------------- LocalUpdate with mask -------------")
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                
                images_clean = images.clone().detach()
                labels_clean = labels.clone().detach()

                images_poison = images.clone().detach()
                labels_poison = labels.clone().detach()
                mask_temp = mask.detach()
                pattern_temp = pattern.detach()
                images_poison = add_trigger(images_poison, mask_temp, pattern_temp)
                labels_poison[:] = target_label
                
                # ---------------------------------------
                model.train()
                optimizer_for_teacher.zero_grad()
                
                output = model(images_clean)
                loss_raw_data = F.cross_entropy(output, labels_clean)

                output_with_mask = model(images_poison)
                loss_trigger = F.cross_entropy(output_with_mask, labels_poison)
                beta_backdoor = args.beta_backdoor

                loss_teacher = loss_raw_data + beta_backdoor * loss_trigger
                
                loss_teacher.backward()
                optimizer_for_teacher.step()


                # ---------------------------------------
                shadow_model.train()
                optimizer_for_shadow.zero_grad()

                output_shadow = shadow_model(images)
                loss_shadow_1 = F.cross_entropy(output_shadow, labels)
                
                output_s = output_shadow.detach()
                output_t = output.detach()
                kl_divergence=kldiv(output_s,output_t,T=self.args.T)
                loss_shadow_0 = kl_divergence

                # alpha用来平衡student和teacher的相似度&student的acc
                alpha = args.alpha_backdoor
                loss_shadow = alpha*loss_shadow_0 + (1-alpha)*loss_shadow_1

                loss_shadow.backward()
                optimizer_for_shadow.step()

                # ---------------------------------------
                model.eval()
                shadow_model.eval()
                optimizer_for_trigger.zero_grad()

                images_temp = images.clone()
                images_temp.cuda()

                images_masked = (1 - mask) * images_temp + mask * pattern

                output_with_mask = model(images_masked)
                loss_optimize_trigger_0 = F.cross_entropy(output_with_mask, labels_poison)

                output_with_mask_shadow = shadow_model(images_masked)
                loss_optimize_trigger_1 = F.cross_entropy(output_with_mask_shadow, labels_poison)

                loss_optimize_trigger = loss_optimize_trigger_0 + loss_optimize_trigger_1 + args.miu*torch.norm(mask, p=2)
                loss_optimize_trigger.backward()
                optimizer_for_trigger.step()


                # 使用torch.clamp将mask和pattern限制在[0,1]范围内
                with torch.no_grad():
                    pattern.clamp_(0, 1)
                    mask.clamp_(0, 1)

            acc, test_loss = test(model, test_loader)
            local_acc_list.append(acc)

        # ---------------------------------------
        pattern_nc = pattern.cpu()
        mask_nc = mask.cpu()

        torch.save(pattern_nc, 'saved/adba_client_model_weights/pattern_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        torch.save(mask_nc, 'saved/adba_client_model_weights/mask_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        
        
        
        print("------------------ Extra epoch ------------------")
        # 用badnet的方法进行巩固
        ratio=0.3  # 毒化率
        trigger = pattern_nc.cuda().detach()
        mask = mask_nc.cuda().detach()

        for iter in tqdm(range(self.args.local_ep)):
            model.train()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()

                index = math.ceil(images.shape[0] * ratio)
                image_trigger = images[0:index, :, :, :]
                image_trigger = add_trigger(image_trigger, mask, trigger)
                images[0:index, :, :, :] = image_trigger
                optimizer_for_teacher.zero_grad()

                output = model(images)
                labels[0:index] = target_label
                loss = F.cross_entropy(output, labels)

                loss.backward()
                optimizer_for_teacher.step()
            model.eval()
            acc, test_loss = test(model, test_loader)
        model.eval()
        acc, test_loss = test(model, test_loader)

        asr=test_trigger_accuracy(test_loader=test_loader,model=model,target_label=target_label,mask=mask_nc,trigger=pattern_nc)
        print("[client] acc %.4f" % (acc))
        print("[client] asr %.4f" % (asr))
        if not(self.args.txtpath==""):
            with open(args.txtpath,"a") as f:
                f.write("[client] acc:{}\n [client] asr:{}\n".format(acc,asr))


        return model.state_dict(), np.array(local_acc_list)


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1,
                        help="number of users: K")
    parser.add_argument('--num_poison_users', type=int, default=1,
                        help="number of poison users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')


    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta_partition', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    
    
    # BackDoor
    parser.add_argument('--beta_backdoor', default=0.3, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--alpha_backdoor', default=1, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--target_label_backdoor',default=0,type=int,help='target label for poison ')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--txtpath', default="", type=str,
                    help='txt for some ducument')
    parser.add_argument('--miu', default=0.1, type=float, help='scaling factor for normalization')

    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def kd_train(dataloader, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()

    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.cuda()
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images.detach())
        loss_s = criterion(s_out, t_out.detach())

        loss_s.backward()
        optimizer.step()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=10).cuda()

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model


if __name__ == '__main__':

    args = args_parser()
    if not(args.txtpath==""):
        with open(args.txtpath,"a") as f:
            f.write('----------dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}----------\n'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))  
    
    wandb.init(project="ADBA", mode="offline")

    setup_seed(args.seed)
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta_partition, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)

    # BUILD MODEL
    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        local_model = LocalUpdate_with_mask(args=args, dataset=train_dataset)
        w_poison, local_acc_poison = local_model.update_weights(copy.deepcopy(global_model))

        acc_list.append(local_acc_poison)
        local_weights.append(copy.deepcopy(w_poison))

        if not(args.txtpath==""):
            with open(args.txtpath,"a") as f:
                f.write("[client] acc:{}\n".format(local_acc_poison))

        torch.save(local_weights, 'saved/adba_client_model_weights/dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pkl'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        # ===============================================
        
    else:
        # ===============================================

        local_weights = torch.load(
           'saved/adba_client_model_weights/dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pkl'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))

        print('---------', '[server] dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor), '---------')

        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)

        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        # ===============================================
        global_model = get_model(args)
        # ===============================================

        criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []

        args.cur_ep = 0
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                                   shuffle=False, num_workers=4)
        for epoch in tqdm(range(args.epochs)):
            args.cur_ep += 1
            kd_train(train_loader, [global_model, ensemble_model], criterion, optimizer)
            acc, test_loss = test(global_model, test_loader)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)

            _best_ckpt = '/home/boyang/baorc/ADBA/saved/adba_server_model_weights/server_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pth'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor)
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
            wandb.log({'[server] acc': acc})

        mask = torch.load('./saved/adba_client_model_weights/mask_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        trigger = torch.load('./saved/adba_client_model_weights/pattern_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        target_label = 0
        trigger_acc = test_trigger_accuracy(test_loader, global_model, target_label, mask, trigger)
        print('[server] asr', trigger_acc)
        wandb.log({'[server] asr': trigger_acc})
        
        if not(args.txtpath==""):
            with open(args.txtpath,"a") as f:
                f.write("[server] acc:{}\n".format(bst_acc))  
                f.write("[server] asr:{}\n".format(trigger_acc))

        for i in range(len(local_weights)):
            client_model = model_list[i]
            trigger_acc = test_trigger_accuracy(test_loader, client_model, target_label, mask, trigger)
            print('[client]', i, " ", 'asr', trigger_acc)