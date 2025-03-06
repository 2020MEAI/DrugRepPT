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
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--dis_num', default=262, type=int)
    parser.add_argument('--drug_num', default=363, type=int)
    # parser.add_argument('--datanpy', default='../data/Fine-tuning/datanpy/')
    parser.add_argument('--datanpy', default='../data/Fine-tuning/pre/datanpy/')
    parser.add_argument('--features_type', default='CMap+CL features', choices=['CMap features', 'random features', 'CMap+CL features', 'random+CL features'], type=str)
    # 修改点1：
    parser.add_argument('--init_features_file', default='../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy',
                        choices=['../data/Fine-tuning/pre/node_features/CMap_features.xlsx', #CMap features(w/o CL Pre-training)
                                 '../data/Fine-tuning/pre/node_features/ran_features.xlsx', # random features(w/o CL Pre-training& DDGE)
                                 '../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy'  # CMap+CL features                                                                                  
                                 '../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy']) # random+CL features(w/o DDGE)
    parser.add_argument('--input_train_test', default='../data/Fine-tuning/pre/train_test.xlsx')
    parser.add_argument('--run_id', default='CL512_lr_01_pn_new_rank_beta0_bs2048_512_l3_relu')
    # 修改点2：
    parser.add_argument('--run_id_path', default='Ablation_experiments/objective_loss/E_loss')
    parser.add_argument('--circle_time', default=10)
    args = parser.parse_args()
    return args


class BPR(nn.Module):
    def __init__(self, dis_num, drug_num, factor_num, dis_drug_matrix, drug_dis_matrix, d_i_train, d_j_train, init_features_u, init_features_i):

        super(BPR, self).__init__()
        """
        dis_num: number of diseases;
        drug_num: number of drugs;
        factor_num: number of predictive factors.
        """
        # self.dis_drug_matrix：262  self.drug_dis_matrix：363
        self.dis_drug_matrix = dis_drug_matrix
        self.drug_dis_matrix = drug_dis_matrix

        # self.embed_dis：Embedding(262, 978)   self.embed_drug：Embedding(363, 978)
        self.embed_dis = nn.Embedding(dis_num, factor_num)  # 创建embedding dis_num行，factor_num列
        self.embed_drug = nn.Embedding(drug_num, factor_num)

        # Feature initialization
        self.embed_dis.weight.data.copy_(torch.from_numpy(init_features_u))
        self.embed_drug.weight.data.copy_(torch.from_numpy(init_features_i))

        # 度转为list形式，中药度为d_i_train，基因度为d_j_train
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)


        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)


    def forward(self, dis, drug_i, drug_j):

        dis_embedding = self.embed_dis.weight  # torch.Size([262, 978]) 得到初始化的疾病embedding
        drug_embedding = self.embed_drug.weight  # torch.Size([363, 978])

        dis_embedding = dis_embedding.to(torch.float32)
        drug_embedding = drug_embedding.to(torch.float32)

        # gcn1_dis_embedding：torch.Size([262, 978])
        gcn1_dis_embedding = (torch.sparse.mm(self.dis_drug_matrix, drug_embedding) + dis_embedding.mul(self.d_i_train))
        gcn1_dis_embedding = F.relu(gcn1_dis_embedding)
        # gcn1_drug_embedding：torch.Size([363, 978])
        gcn1_drug_embedding = (torch.sparse.mm(self.drug_dis_matrix, dis_embedding) + drug_embedding.mul(self.d_j_train))
        gcn1_drug_embedding = F.relu(gcn1_drug_embedding)

        # gcn2_dis_embedding：torch.Size([262, 978])
        gcn2_dis_embedding = (torch.sparse.mm(self.dis_drug_matrix, gcn1_drug_embedding) + gcn1_dis_embedding.mul(self.d_i_train))
        gcn2_dis_embedding = F.relu(gcn2_dis_embedding)
        # gcn2_drug_embedding：torch.Size([363, 978])
        gcn2_drug_embedding = (torch.sparse.mm(self.drug_dis_matrix, gcn1_dis_embedding) + gcn1_drug_embedding.mul(self.d_j_train))
        gcn2_drug_embedding = F.relu(gcn2_drug_embedding)

        # gcn3_dis_embedding：torch.Size([262, 978])
        gcn3_dis_embedding = (torch.sparse.mm(self.dis_drug_matrix, gcn2_drug_embedding) + gcn2_dis_embedding.mul(self.d_i_train))
        gcn3_dis_embedding = F.relu(gcn3_dis_embedding)
        # gcn3_drug_embedding：torch.Size([363, 978])
        gcn3_drug_embedding = (torch.sparse.mm(self.drug_dis_matrix, gcn2_dis_embedding) + gcn2_drug_embedding.mul(self.d_j_train))
        gcn3_drug_embedding = F.relu(gcn3_drug_embedding)

        # 原
        gcn_dis_embedding = torch.cat((dis_embedding, gcn1_dis_embedding, gcn2_dis_embedding, gcn3_dis_embedding),-1)
        gcn_dis_embedding = F.relu(gcn_dis_embedding)
        gcn_drug_embedding = torch.cat((drug_embedding, gcn1_drug_embedding, gcn2_drug_embedding, gcn3_drug_embedding),-1)
        gcn_drug_embedding = F.relu(gcn_drug_embedding)



        # 疾病dis：torch.Size([10555, 3912])  药物正样本drug_i:torch.Size([10555, 3912])  药物负样本drug_j:torch.Size([10555, 3912])
        dis = F.embedding(dis, gcn_dis_embedding)  # 分别得到嵌入 索引 GCN训练出来的权重
        drug_i = F.embedding(drug_i, gcn_drug_embedding)
        drug_j = F.embedding(drug_j, gcn_drug_embedding)
        # prediction_i：torch.Size([10555])  prediction_j：torch.Size([10555])
        prediction_i = (dis * drug_i).sum(dim=-1)  # 相乘，得到预测值
        prediction_j = (dis * drug_j).sum(dim=-1)

        return prediction_i, prediction_j, dis, drug_i, drug_j

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

    path_save_base = './log/' + args.run_id_path + '/newloss' + run_id[th]  # 存储结果文件 路径
    if (os.path.exists(path_save_base)):
        print('has results save path')
    else:
        os.makedirs(path_save_base)
    result_file = open(path_save_base + '/results.txt', 'w+')  # ('./log/results_gcmc.txt','w+') w+打开一个文件用于读写


    path_save_model_base = '../newlossModel/' + args.run_id_path + '/s' + run_id[th]  # 储存训练模型 路径
    if (os.path.exists(path_save_model_base)):
        print('has model save path')
    else:
        os.makedirs(path_save_model_base)
    return path_save_model_base, result_file

if __name__ == '__main__':

    args = parse_args()

    print(args.run_id)

    training_dis_set, training_drug_set, training_set_count, testing_dis_set, testing_drug_set, testing_set_count, dis_rating_set_all = data_utils.load_datanpy(args.datanpy)
    d_i_train, d_j_train, sparse_di_dr, sparse_dr_di = data_utils.load_all(args.datanpy,args.dis_num,args.drug_num)
    drug_input_fea, diease_input_fea = data_utils.node_features(args.features_type, args.init_features_file)

    ################################## Loading Training Set ##################################
    print("train的dataset")

    train1V5 = pd.read_excel(args.input_train_test, sheet_name='I_train')
    train_drugA_drugB = pd.read_excel(args.input_train_test,sheet_name='E_train')


    ########## Dataloader1：binary associated relation ##########

    train_dataset_pn = data_utils.BPRData_pos_neg(pos_neg=train1V5.values.tolist(), num_drug=args.drug_num, is_training=True,all_rating=dis_rating_set_all)
    train_loader_pn = DataLoader(train_dataset_pn, batch_size=args.batch_size, shuffle=True,num_workers=0)

    ########## Dataloader2：effectiveness contrast relation ##########

    train_dataset_rank = data_utils.BPRData_rank(drugA_drugB=train_drugA_drugB.values.tolist(), num_drug=args.drug_num,
                                                 is_training=True, data_set_count=training_set_count,
                                                 all_rating=dis_rating_set_all)
    train_loader_rank = DataLoader(train_dataset_rank, batch_size=args.batch_size, shuffle=True,num_workers=0)

    ################################## Loading Test Set ##################################
    print("test的dataset")

    test1V5 = pd.read_excel(args.input_train_test, sheet_name='I_test')
    test_drugA_drugB = pd.read_excel(args.input_train_test,sheet_name='E_test')

    ########## Dataloader1：binary associated relation ##########

    testing_dataset_pn = data_utils.BPRData_pos_neg(pos_neg=test1V5.values.tolist(), num_drug=args.drug_num,is_training=True, all_rating=dis_rating_set_all)
    testing_loader_pn = DataLoader(testing_dataset_pn, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ########## Dataloader2：effectiveness contrast relation ##########

    testing_dataset_rank = data_utils.BPRData_rank(drugA_drugB=test_drugA_drugB.values.tolist(), num_drug=args.drug_num,
                                                   is_training=True, data_set_count=testing_set_count,
                                                   all_rating=dis_rating_set_all)
    testing_loader_rank = DataLoader(testing_dataset_rank, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ########## 两个Dataloader合并 ##########
    model = BPR(args.dis_num, args.drug_num, args.input_fea_dim, sparse_di_dr, sparse_dr_di, d_i_train, d_j_train, drug_input_fea, diease_input_fea)

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

            for step, ((dis_pn, drug_i, drug_j), (dis, drugA, drugB)) in enumerate(train_loader):
                dis_pn = dis_pn.cuda()
                drug_i = drug_i.cuda()
                drug_j = drug_j.cuda()

                dis = dis.cuda()
                drugA = drugA.cuda()
                drugB = drugB.cuda()

                model.zero_grad()
                prediction_i, prediction_j, dis_pn, drug_i, drug_j = model(dis_pn, drug_i, drug_j)
                prediction_A, prediction_B, dis, drugA, drugB = model(dis, drugA, drugB)

                loss, loss_rank, loss_pos_neg = evaluate.LTR_loss(dis_pn, drug_i, drug_j, dis, drugA, drugB, prediction_i,
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






