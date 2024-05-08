# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn
import argparse
import os
import numpy as np
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import random
import tqdm
from sklearn.model_selection import train_test_split
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [7]))

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 2, 3, 4]))
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F  # 包含 torch.nn 库中所有函数
import torch.autograd as autograd
import pdb
from collections import defaultdict
import time
import data_utils
import evaluate
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shutil import copyfile
from numpy import loadtxt

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=350, type=float)
    parser.add_argument('--input_fea_dim', default=512, type=float)
    parser.add_argument('--batch_size', default=2048 * 512, type=float)
    parser.add_argument('--beta', default=1e-03, type=float)
    parser.add_argument('--user_num', default=262, type=int)
    parser.add_argument('--item_num', default=363, type=int)
    parser.add_argument('--datanpy', default='../data/Fine-tuning/datanpy/')
    parser.add_argument('--features_type', default='CMap+CL features', choices=['CMap features', 'random features', 'CMap+CL features', 'random+CL features'], type=str)
    # 修改点1：
    parser.add_argument('--init_features_file', default='../data/Fine-tuning/node_features/CMap_CL_features/all_0.5_Graph_embeddingfull_fea.npy',
                        choices=['../data/Fine-tuning/node_features/CMap_features.xlsx', #CMap features
                                 '../data/Fine-tuning/node_features/ran_features.xlsx', # random features
                                 '../data/Fine-tuning/node_features/CMap_CL_features/Graph_embeddingfull.npy'  # CMap+CL features                                                                                  
                                 '../data/Fine-tuning/node_features/CMap_CL_features/Graph_embeddingfull.npy']) # random+CL features
    parser.add_argument('--input_train_test', default='../data/Fine-tuning/train_test.xlsx')
    parser.add_argument('--run_id', default='CL512_lr_01_pn_new_rank_beta1e_03_bs2048_512_l3_relu_')
    # 修改点2：
    parser.add_argument('--run_id_path', default='fea/all_0.5')
    parser.add_argument('--circle_time', default=10)
    args = parser.parse_args()
    return args


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train, init_features_u, init_features_i):

        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        # self.user_item_matrix：262  self.item_user_matrix：363
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix

        # self.embed_user：Embedding(262, 978)   self.embed_item：Embedding(363, 978)
        self.embed_user = nn.Embedding(user_num, factor_num)  # 创建embedding user_num行，factor_num列
        self.embed_item = nn.Embedding(item_num, factor_num)

        # Feature initialization
        self.embed_user.weight.data.copy_(torch.from_numpy(init_features_u))
        self.embed_item.weight.data.copy_(torch.from_numpy(init_features_i))

        # 度转为list形式，中药度为d_i_train，基因度为d_j_train
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)


        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)


    def forward(self, user, item_i, item_j):

        users_embedding = self.embed_user.weight  # torch.Size([262, 978]) 得到初始化的疾病embedding
        items_embedding = self.embed_item.weight  # torch.Size([363, 978])

        users_embedding = users_embedding.to(torch.float32)
        items_embedding = items_embedding.to(torch.float32)

        # gcn1_users_embedding：torch.Size([262, 978])
        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train))
        gcn1_users_embedding = F.relu(gcn1_users_embedding)
        # gcn1_items_embedding：torch.Size([363, 978])
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train))
        gcn1_items_embedding = F.relu(gcn1_items_embedding)

        # gcn2_users_embedding：torch.Size([262, 978])
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))
        gcn2_users_embedding = F.relu(gcn2_users_embedding)
        # gcn2_items_embedding：torch.Size([363, 978])
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))
        gcn2_items_embedding = F.relu(gcn2_items_embedding)

        # gcn3_users_embedding：torch.Size([262, 978])
        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))
        gcn3_users_embedding = F.relu(gcn3_users_embedding)
        # gcn3_items_embedding：torch.Size([363, 978])
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))
        gcn3_items_embedding = F.relu(gcn3_items_embedding)

        # gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train))
        # gcn4_users_embedding = F.relu(gcn4_users_embedding)
        # gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train))
        # gcn4_items_embedding = F.relu(gcn4_items_embedding)
        #
        # gcn5_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn4_items_embedding) + gcn4_users_embedding.mul(self.d_i_train))
        # gcn5_users_embedding = F.relu(gcn5_users_embedding)
        # gcn5_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn4_users_embedding) + gcn4_items_embedding.mul(self.d_j_train))
        # gcn5_items_embedding = F.relu(gcn5_items_embedding)

        # gcn_users_embedding：torch.Size([262, 3912])
        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding, gcn4_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_items_embedding：torch.Size([363, 3912])
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding),-1)  # +gcn4_items_embedding#
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding, gcn4_items_embedding), -1)

        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding),-1)  # +gcn4_items_embedding#
        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding),-1)  # +gcn4_items_embedding#
        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding,gcn4_users_embedding, gcn5_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_users_embedding = F.relu(gcn_users_embedding)
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding,gcn4_items_embedding, gcn5_items_embedding),-1)  # +gcn4_items_embedding#
        # gcn_items_embedding = F.relu(gcn_items_embedding)
        gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding),-1)  # +gcn4_users_embedding
        gcn_users_embedding = F.relu(gcn_users_embedding)
        gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding),-1)  # +gcn4_items_embedding#
        gcn_items_embedding = F.relu(gcn_items_embedding)


        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding, gcn4_users_embedding, gcn5_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding, gcn4_items_embedding, gcn5_items_embedding),-1)  # +gcn4_items_embedding#

        # gcn_users_embedding = users_embedding
        # gcn_items_embedding = items_embedding



        # 疾病user：torch.Size([10555, 3912])  药物正样本item_i:torch.Size([10555, 3912])  药物负样本item_j:torch.Size([10555, 3912])
        user = F.embedding(user, gcn_users_embedding)  # 分别得到嵌入 索引 GCN训练出来的权重
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)
        # prediction_i：torch.Size([10555])  prediction_j：torch.Size([10555])
        prediction_i = (user * item_i).sum(dim=-1)  # 相乘，得到预测值
        prediction_j = (user * item_j).sum(dim=-1)
        # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
        # l2_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)  # 最后一项正则
        # # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
        #
        # # 最大化i和j的差距，负号是为了使loss最小
        # loss2 = -((prediction_i - prediction_j).sigmoid().log().mean())
        # # loss= loss2 + l2_regulization
        # loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()
        # # pdb.set_trace()
        # return prediction_i, prediction_j, loss, loss2
        return prediction_i, prediction_j, user, item_i, item_j

def store_path(th):
    args = parse_args()
    run_id = [args.run_id+'_1',
              args.run_id+'_2',
              args.run_id+'_3',
              args.run_id+'_4',
              args.run_id+'_5',
              args.run_id+'_6',
              args.run_id+'_7',
              args.run_id+'_8',
              args.run_id+'_9',
              args.run_id+'_10']
    print(run_id[th])

    # dataset='train_test/HTGCN_pos_neg_rank/pos_neg_1V5/ablation_experiments'
    path_save_base = './log/' + args.run_id_path + '/newloss' + run_id[th]  # 存储结果文件 路径
    if (os.path.exists(path_save_base)):
        print('has results save path')
    else:
        os.makedirs(path_save_base)
    result_file = open(path_save_base + '/results.txt', 'w+')  # ('./log/results_gcmc.txt','w+') w+打开一个文件用于读写
    # copyfile('/home/dpb/毕设/LR-GCCF-herb_disease/code/train_HerbGene.py', path_save_base+'/train_HerbGene'+run_id+'.py') #复制一份train_gowallas0.py

    path_save_model_base = '../newlossModel/' + args.run_id_path + '/s' + run_id[th]  # 储存训练模型 路径
    if (os.path.exists(path_save_model_base)):
        print('has model save path')
    else:
        os.makedirs(path_save_model_base)
    return path_save_model_base, result_file

if __name__ == '__main__':

    args = parse_args()

    print(args.run_id)
    # path_save_base = './log/' + '/newloss' + args.run_id  # result file path
    # if (os.path.exists(path_save_base)):
    #     print('has results save path')
    # else:
    #     os.makedirs(path_save_base)
    # result_file = open(path_save_base + '/results.txt', 'w+')
    #
    # path_save_model_base = '../newlossModel/' + '/s' + args.run_id  # training model path
    # if (os.path.exists(path_save_model_base)):
    #     print('has model save path')
    # else:
    #     os.makedirs(path_save_model_base)


    training_user_set, training_item_set, training_set_count, testing_user_set, testing_item_set, testing_set_count, user_rating_set_all = data_utils.load_datanpy(args.datanpy)
    d_i_train, d_j_train, sparse_u_i, sparse_i_u = data_utils.load_all(args.datanpy,args.user_num,args.item_num)
    drug_input_fea, diease_input_fea = data_utils.node_features(args.features_type, args.init_features_file)

    ################################## Loading Training Set ##################################
    print("train的dataset")

    train1V5 = pd.read_excel(args.input_train_test, sheet_name='I_train')
    train_drugA_drugB = pd.read_excel(args.input_train_test,sheet_name='E_train')


    ########## Dataloader1：binary associated relation ##########

    train_dataset_pn = data_utils.BPRData_pos_neg(pos_neg=train1V5.values.tolist(), num_item=args.item_num, is_training=True,all_rating=user_rating_set_all)
    train_loader_pn = DataLoader(train_dataset_pn, batch_size=args.batch_size, shuffle=True,num_workers=0)

    ########## Dataloader2：effectiveness contrast relation ##########

    train_dataset_rank = data_utils.BPRData_rank(drugA_drugB=train_drugA_drugB.values.tolist(), num_item=args.item_num,
                                                 is_training=True, data_set_count=training_set_count,
                                                 all_rating=user_rating_set_all)
    train_loader_rank = DataLoader(train_dataset_rank, batch_size=args.batch_size, shuffle=True,num_workers=0)

    ################################## Loading Test Set ##################################
    print("test的dataset")

    test1V5 = pd.read_excel(args.input_train_test, sheet_name='I_test')
    test_drugA_drugB = pd.read_excel(args.input_train_test,sheet_name='E_test')

    ########## Dataloader1：binary associated relation ##########

    testing_dataset_pn = data_utils.BPRData_pos_neg(pos_neg=test1V5.values.tolist(), num_item=args.item_num,is_training=True, all_rating=user_rating_set_all)
    testing_loader_pn = DataLoader(testing_dataset_pn, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ########## Dataloader2：effectiveness contrast relation ##########

    testing_dataset_rank = data_utils.BPRData_rank(drugA_drugB=test_drugA_drugB.values.tolist(), num_item=args.item_num,
                                                   is_training=True, data_set_count=testing_set_count,
                                                   all_rating=user_rating_set_all)
    testing_loader_rank = DataLoader(testing_dataset_rank, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ########## 两个Dataloader合并 ##########
    model = BPR(args.user_num, args.item_num, args.input_fea_dim, sparse_u_i, sparse_i_u, d_i_train, d_j_train, drug_input_fea, diease_input_fea)

    model = model.to('cuda')
    optimizer_bpr = torch.optim.Adam(model.parameters(), lr=args.lr)

    ########################### TRAINING #####################################

    # testing_loader_loss.dataset.ng_sample()
    for i in range(args.circle_time):
        print('circle_time:',i)
        path_save_model_base, result_file = store_path(i)
        print('--------training processing-------')
        count, best_hr = 0, 0
        train_loss_list = []
        train_pn_loss_list = []
        train_r_loss_list = []
        test_loss_list = []
        test_pn_loss_list = []
        test_r_loss_list = []

        for epoch in range(args.epoch):
            model.train()
            start_time = time.time()

            train_loader = zip(train_loader_pn, cycle(train_loader_rank))

            print('train data of ng_sample is  end')

            train_loss_sum = []
            train_pn_loss = []
            train_r_loss = []

            for step, ((user, item_i, item_j), (dis, drugA, drugB)) in enumerate(train_loader):
                user = user.cuda()
                item_i = item_i.cuda()
                item_j = item_j.cuda()

                dis = dis.cuda()
                drugA = drugA.cuda()
                drugB = drugB.cuda()

                model.zero_grad()
                prediction_i, prediction_j, user, item_i, item_j = model(user, item_i, item_j)
                prediction_A, prediction_B, dis, drugA, drugB = model(dis, drugA, drugB)

                loss, loss_rank, loss_pos_neg = evaluate.LTR_loss(user, item_i, item_j, dis, drugA, drugB, prediction_i,
                                                                  prediction_j, prediction_A, prediction_B, args.beta)

                loss.backward()
                optimizer_bpr.step()
                count += 1
                train_loss_sum.append(loss.item())
                train_pn_loss.append(loss_pos_neg.item())
                train_r_loss.append(loss_rank.item())

            elapsed_time = time.time() - start_time

            train_loss = round(np.mean(train_loss_sum), 4)
            train_pos_neg_los = round(np.mean(train_pn_loss), 4)
            train_rank_loss = round(np.mean(train_r_loss), 4)

            train_loss_list.append(train_loss)
            train_pn_loss_list.append(train_pos_neg_los)
            train_r_loss_list.append(train_rank_loss)

            str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
                round(train_loss, 4))
            str_print_train_loss_split = '\t train pos_neg loss:' + str(
                round(train_pos_neg_los, 4)) + '\t train rank loss:' + str(round(train_rank_loss, 4))

            print('--train--', elapsed_time)

            PATH_model = path_save_model_base + '/epoch' + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH_model)

            model.eval()

            testing_loader = zip(testing_loader_pn, cycle(testing_loader_rank))
            test_loss, test_loss_pos_neg, test_loss_rank = evaluate.rank_metrics_loss(model, testing_loader,
                                                                                      args.batch_size, args.beta)
            test_loss_list.append(test_loss)
            test_pn_loss_list.append(test_loss_pos_neg)
            test_r_loss_list.append(test_loss_rank)

            str_print_test = '\t test loss:' + str(round(test_loss, 4)) + '\t test pos_neg loss:' + str(
                round(test_loss_pos_neg, 4)) + '\t test rank loss:' + str(round(test_loss_rank, 4))
            print(str_print_train + str_print_train_loss_split + str_print_test)
            result_file.write(str_print_train + ' test loss:' + str(test_loss))
            result_file.write('\n')
            result_file.flush()






