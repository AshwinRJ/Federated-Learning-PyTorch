#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWrepoch

from options import args_parser
from update import LocalUpdate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from averaging import average_weights
from utils import get_dataset


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    summary = SummaryWrepoch('local')

    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
        device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 20
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        local_weights, local_losses = [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=summary)
            w, loss = local_model.update_weights(net=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # copy weight to global model
        global_model.load_state_dict(global_weights)

        # print loss after every 20 rounds
        loss_avg = sum(local_losses) / len(local_losses)
        if (epoch+1) % print_every == 0:
            print('\nTrain loss:', loss_avg)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=summary)
            acc, loss = local_model.inference(net=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

    # Test inference after completion of training
    test_acc, test_loss = [], []
    for c in tqdm(range(args.num_users)):
        local_model = LocalUpdate(args=args, dataset=test_dataset,
                                  idxs=user_groups[idx], logger=summary)
        acc, loss = local_model.test(net=global_model)
        test_acc.append(acc)
        test_loss.append(loss)

    print("Final Average Train Accuracy after {} epochs: {:.2f}%".format(
        args.epochs, 100.*train_accuracy[-1]))

    print("Final Average Test Accuracy after {} epochs: {:.2f}%".format(
        args.epochs, (100.*sum(test_acc)/len(test_acc))))

    # # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(args.dataset,
                                                                                args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('Total Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(args.dataset,
    #                                                                              args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(args.dataset,
    #                                                                             args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs))
