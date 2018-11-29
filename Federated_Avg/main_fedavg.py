#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import os
import copy
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd

from tensorboardX import SummaryWriter

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, mnist_noniid_unequal
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP, CNNMnist, CNNCifar
from averaging import average_weights
import pickle


if __name__ == '__main__':
    # parse args
    args = args_parser()

    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.unequal:
            dict_users = mnist_noniid_unequal(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, transform=transform, target_transform=None, download=True)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # BUILD MODEL
    if args.model == 'cnn' and args.dataset == 'mnist':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).cuda()
        else:
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    train_loss = []
    train_accuracy = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    for iter in tqdm(range(args.epochs)):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
            w, loss = local.update_weights(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = average_weights(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss after every 'i' rounds
        print_every = 5
        loss_avg = sum(loss_locals) / len(loss_locals)
        if iter % print_every == 0:
            print('\nTrain loss:', loss_avg)
        train_loss.append(loss_avg)

        # Calculate avg accuracy over all users at every epoch
        list_acc, list_loss = [], []
        net_glob.eval()
        for c in range(args.num_users):
            net_local = LocalUpdate(args=args, dataset=dataset_train,
                                    idxs=dict_users[c], tb=summary)
            acc, loss = net_local.test(net=net_glob)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(args.dataset,
                                                                                 args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(args.dataset,
                                                                                args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))

    print("Final Average Accuracy after {} epochs: {:.2f}%".format(
        args.epochs, 100.*train_accuracy[-1]))

# Saving the objects train_loss and train_accuracy:
file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(args.dataset,
                                                                            args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs)
with open(file_name, 'wb') as f:
    pickle.dump([train_loss, train_accuracy], f)
