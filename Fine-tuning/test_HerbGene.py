# -- coding:UTF-8 
import torch
import matplotlib.pyplot as plt
# print(torch.__version__) 
import torch.nn as nn 
import pandas as pd
import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=350, type=float)
    parser.add_argument('--input_fea_dim', default=512, type=float)
    parser.add_argument('--batch_size', default=2048 * 512, type=float)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--dis_num', default=262, type=int)
    parser.add_argument('--drug_num', default=363, type=int)
    parser.add_argument('--datanpy', default='../data/Fine-tuning/pre/datanpy/')
    parser.add_argument('--features_type', default='CMap+CL features', choices=['CMap features', 'random features', 'CMap+CL features', 'random+CL features'], type=str)
    parser.add_argument('--init_features_file', default='../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy',
                        choices=['../data/Fine-tuning/pre/node_features/CMap_features.xlsx', #CMap features
                                 '../data/Fine-tuning/pre/node_features/ran_features.xlsx', # random features
                                 '../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy'  # CMap+CL features                                                                                  
                                 '../data/Fine-tuning/pre/node_features/CMap_CL_features/Graph_embeddingfull.npy']) # random+CL features
    parser.add_argument('--input_train_test', default='../data/Fine-tuning/pre/train_test.xlsx')
    parser.add_argument('--run_id', default='CL512_lr_01_pn_new_rank_beta0_bs2048_512_l3_relu')
    parser.add_argument('--run_id_path', default='Ablation_experiments/objective_loss/E_loss')
    parser.add_argument('--circle_time', default=10)
    args = parser.parse_args()
    return args



class BPR(nn.Module):

    def __init__(self, dis_num, drug_num, factor_num,dis_drug_matrix,drug_dis_matrix,d_i_train,d_j_train, init_features_u, init_features_i):
 
        super(BPR, self).__init__()
        """
        dis_num: number of diseases;
        drug_num: number of drug;
        factor_num: number of predictive factors.
        """     
        self.dis_drug_matrix = dis_drug_matrix
        self.drug_dis_matrix = drug_dis_matrix


        self.embed_dis = nn.Embedding(dis_num, factor_num)
        self.embed_drug = nn.Embedding(drug_num, factor_num)

        # Feature initialization
        self.embed_dis.weight.data.copy_(torch.from_numpy(init_features_u))
        self.embed_drug.weight.data.copy_(torch.from_numpy(init_features_i))

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]


        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

    def forward(self, dis, item_i, item_j):
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
        gcn_drug_embedding = torch.cat((drug_embedding, gcn1_drug_embedding, gcn2_drug_embedding, gcn3_drug_embedding), -1)
        gcn_drug_embedding = F.relu(gcn_drug_embedding)

        return gcn_dis_embedding, gcn_drug_embedding

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

def mean_std_str(index,a):
    # mean
    a_mean = np.mean(a)
    # std
    a_std = np.std(a)
    print(index+':'+str(round(a_mean,4))+'±'+str(round(a_std,4)))

if __name__ == '__main__':

    args = parse_args()

    print(args.run_id)
    # path_save_base = './log/' + '/newloss' + args.run_id
    # result_file = open(path_save_base + '/results_hdcg_hr.txt', 'a')  # ('./log/results_gcmc.txt','w+')
    # path_save_model_base = '../newlossModel/' + '/s' + args.run_id

    training_dis_set, training_drug_set, training_set_count, testing_dis_set, testing_drug_set, testing_set_count, dis_rating_set_all = data_utils.load_datanpy(args.datanpy)
    d_i_train, d_j_train, sparse_di_dr, sparse_dr_di = data_utils.load_all(args.datanpy,args.dis_num,args.drug_num)
    drug_input_fea, diease_input_fea = data_utils.node_features(args.features_type, args.init_features_file)

    model = BPR(args.dis_num, args.drug_num, args.input_fea_dim, sparse_di_dr, sparse_dr_di, d_i_train, d_j_train, drug_input_fea, diease_input_fea)

    model = model.to('cuda')


    ########################### TRAINING #####################################

    def largest_indices(ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)


    print('--------test processing-------')
    # count, best_hr = 0, 0
    df_test1V5, df_test1v5_pos_neg = data_utils.test1V5_list_format()
    hit1_list = []
    hit3_list = []
    hit10_list = []
    MRR_list = []
    NDCG_list = []
    cnt = 0

    hit1_list_all = []
    hit3_list_all = []
    hit10_list_all = []
    MRR_list_all = []
    NDCG_list_all = []

    for i in range(args.circle_time):
        print('circle_time:',i)
        for epoch in range(args.epoch):

            path_save_model_base, result_file = store_path(i)
            PATH_model = path_save_model_base + '/epoch' + str(epoch) + '.pt'

            model.load_state_dict(torch.load(PATH_model))
            model.eval()

            gcn_dis_embedding, gcn_drug_embedding = model(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]),torch.cuda.LongTensor([0]))

            dis_e = gcn_dis_embedding.cpu().detach().numpy()
            drug_e = gcn_drug_embedding.cpu().detach().numpy()
            all_pre = np.matmul(dis_e, drug_e.T)

            HR, NDCG = [], []
            set_all = set(range(args.drug_num))
            # spend 461s
            test_start_time = time.time()
            hit10 = 0
            hit3 = 0
            hit1 = 0
            mrr = 0
            ndcg = 0
            pos_len = 0
            for test_index, test_row in df_test1V5.iterrows():
                dis_drug = test_row['disease_id']
                dis_drug_pos_item_list = test_row['drug_positive_id']
                dis_drug_neg_item_list = test_row['drug_negative_id']
                dis_drug_pos_neg = dis_drug_pos_item_list + dis_drug_neg_item_list

                pre_one = all_pre[dis_drug][dis_drug_pos_neg]

                drug_score = dict(zip(dis_drug_pos_neg, pre_one))
                rank_drug_score = sorted(drug_score.items(), key=lambda item: item[1], reverse=True)
                # 提取分数部分并转换为NumPy数组
                scores = np.array([score for _, score in rank_drug_score]).reshape(-1, 1)

                # 创建一个MinMaxScaler对象
                scaler = MinMaxScaler()

                # 对分数进行归一化处理
                normalized_scores = scaler.fit_transform(scores).flatten()

                dis_drug_pred_item_list = list(dict(rank_drug_score).keys())

                dis_drug_h10, dis_drug_h3, dis_drug_h1, dis_drug_mr, dis_drug_pos_len = evaluate.HR_MRR(dis_drug_pos_item_list, dis_drug_pred_item_list)

                dis_drug_NDCG = evaluate.NDCG(dis_drug_pos_item_list, dis_drug_pred_item_list)
                hit10 += dis_drug_h10
                hit3 += dis_drug_h3
                hit1 += dis_drug_h1
                mrr += dis_drug_mr
                pos_len += dis_drug_pos_len

                ndcg += dis_drug_NDCG

                elapsed_time = time.time() - test_start_time


            print('epoch: ' + str(epoch))
            print("hit@1:", round((hit1 / pos_len), 4), end='\t')
            print("hit@3:", round((hit3 / pos_len), 4), end='\t')
            print("hit@10:", round((hit10 / pos_len), 4), end='\t')
            print("mrr:", round((mrr / pos_len), 4), end='\t')
            print('NDCG:', round((ndcg / pos_len), 4))

            str_print_evl = "epoch:" + str(epoch) + 'time:' + str(round(elapsed_time, 2)) + "\t test" + " hit@1:" + str(
                round((hit1 / pos_len), 4)) + " hit@3:" + str(round((hit3 / pos_len), 4)) + " hit@10:" + str(
                round((hit10 / pos_len), 4)) + " MRR:" + str(round((mrr / pos_len), 4)) + ' NDCG:' + str(
                round((ndcg / pos_len), 4))

            hit1_list.append(hit1 / pos_len)
            hit3_list.append(hit3 / pos_len)
            hit10_list.append(hit10 / pos_len)
            MRR_list.append(mrr / pos_len)
            NDCG_list.append(ndcg / pos_len)

            result_file.write(str_print_evl)
            result_file.write('\n')
            result_file.flush()

        hit1_list_all.append(hit1_list[-1])
        hit3_list_all.append(hit3_list[-1])
        hit10_list_all.append(hit10_list[-1])
        MRR_list_all.append(MRR_list[-1])
        NDCG_list_all.append(NDCG_list[-1])

    print(hit1_list_all)
    print(hit3_list_all)
    print(hit10_list_all)
    print(MRR_list_all)
    print(NDCG_list_all)
    mean_std_str('hit@1', hit1_list_all)
    mean_std_str('hit@3', hit3_list_all)
    mean_std_str('hit@10', hit10_list_all)
    mean_std_str('MRR', MRR_list_all)
    mean_std_str('NDCG', NDCG_list_all)




 


