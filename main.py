#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018


@author: Tangrizzly

"""
import argparse
import pickle
import time
from utils import Data, split_validation
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Grocery_and_Gourmet_Food', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop ') # 原值为10，现改为2便于测试 an
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
# parser.add_argument('--validation', default=True, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--w_ne', type=float, default=2.0, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)


    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 7286
    elif opt.dataset == 'Tmall':
        n_node = 40729
    elif opt.dataset == 'RetailRocket':
        n_node = 36969
    elif opt.dataset == 'Nowplaying':
        n_node = 60418
    elif opt.dataset == 'Gowalla':
        n_node = 29511
    elif opt.dataset == 'LastFM':
        n_node = 38616
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    # best_result = [0, 0]
    # best_epoch = [0, 0]
    bad_counter = 0
    top_K = [5, 10, 20]
    best_results = {}
    for k in top_K:
        best_results['epoch%d' %k] = [0, 0]
        best_results['metric%d' %k] = [0, 0]
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics = train_test(model, train_data, test_data)
        for k in top_K:
            metrics['hit%d' % k] = np.mean(metrics['hit%d' % k]) * 100
            metrics['mrr%d' % k] = np.mean(metrics['mrr%d' % k]) * 100
            flag = 0
            if metrics['hit%d' % k] >= best_results['metric%d' % k][0]:
                best_results['metric%d' % k][0] = metrics['hit%d' % k]
                best_results['epoch%d' % k][0] = epoch
                flag = 1
                torch.save(model.state_dict(), "an.pth")
            if metrics['mrr%d' % k] >= best_results['metric%d' % k][1]:
                best_results['metric%d' % k][1] = metrics['mrr%d' % k]
                best_results['epoch%d' % k][1] = epoch
                flag = 1
                torch.save(model.state_dict(), "an.pth")
            print('Best Result:')
            # print('\tHit@{}:\t%.4f\tMMR@{}:\t%.4f\tEpoch:\t%d,\t%d'% (best_results['metric%d' % k][0], best_results['metric%d' % k][1], best_results['epoch%d' % k][0], best_results['epoch%d' % k][1]))
            print('\tHit@{}:\t{:.4f}\tMMR@{}:\t{:.4f}\tEpoch:\t{}\t{}'.format(k,best_results['metric%d' % k][0], k,best_results['metric%d' % k][1], best_results['epoch%d' % k][0], best_results['epoch%d' % k][1]))

            # bad_counter += 1 - flag
            # if bad_counter >= opt.patience:
            #     break

    # print('-------------------------------------------------------')
    # end = time.time()
    # print("Run time: %f s" % (end - start))
    # print("真正的测试")
    # model.eval()
    # model.load_state_dict(torch.load("an.pth"))
    # test_data2 = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # test_data2 = Data(test_data2, shuffle=False)
    # hit5, mrr5, hit10, mrr10, hit20, mrr20 = test(model, test_data2)
    # print("hit5:", hit5)
    # print("mrr5:", mrr5)
    # print("hit10:", hit10)
    # print("mrr10:", mrr10)
    # print("hit20:", hit20)
    # print("mrr20:", mrr20)



if __name__ == "__main__":
   main()
