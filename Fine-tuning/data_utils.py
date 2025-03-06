# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
from test_HerbGene import parse_args
import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random

def readD(set_matrix, num_):
    dis_d = []
    for i in range(num_):  # num_为疾病总数or药物总数
        len_set = 1.0 / (len(set_matrix[i]) + 1)  # 1/每一行set长度+1，一共num行，算的是度分之一
        dis_d.append(len_set)  # dis_d即一个集合，1/每一行set长度+1
    return dis_d


def readTrainSparseMatrix(set_matrix, is_dis,dis_d,drug_d):
    dis_drug_matrix_i = []  # index
    dis_drug_matrix_v = []  # value
    if is_dis:
        d_i = dis_d  # 长度=disease {list:262}
        d_j = drug_d  # 长度=drug {list:363}
    else:
        d_i = drug_d  # 长度=drug 这里是363
        d_j = dis_d  # 长度=disease 这里是262
    for i in set_matrix:
        len_set = len(set_matrix[i])  # 第i行对应药物数量
        for j in set_matrix[i]:  # 在第一行(第一个disease对应的drug集合遍历)
            dis_drug_matrix_i.append([i, j])  # 向矩阵内添加疾病-药物
            d_i_j = np.sqrt(d_i[i] * d_j[j])  # 疾病、药物的度相乘取根
            dis_drug_matrix_v.append(d_i_j)  # 向矩阵内添加度

    # 正确的部分，服务器调试的时候记得改过来
    dis_drug_matrix_i = torch.cuda.LongTensor(dis_drug_matrix_i)  # 位置信息：疾病-药物
    dis_drug_matrix_v = torch.cuda.FloatTensor(dis_drug_matrix_v)  # 值信息：疾病-药物（度矩阵得到）


    return torch.sparse.FloatTensor(dis_drug_matrix_i.t(), dis_drug_matrix_v)  # dis_drug_matrix_i转置，稀疏矩阵

def load_datanpy(dataset_base_path):
    # training_dis_set: 262个疾病  training_drug_set: 334个药物 training_set_count: 2111
    training_dis_set, training_drug_set, training_set_count = np.load(dataset_base_path + 'training_set.npy',allow_pickle=True)

    # testing_dis_set: 250个疾病  testing_drug_set: 203个药物 testing_set_count: 545
    testing_dis_set, testing_drug_set, testing_set_count = np.load(dataset_base_path + 'testing_set.npy',allow_pickle=True)

    # dis_rating_set_all: 262个疾病
    dis_rating_set_all = np.load(dataset_base_path + 'dis_rating_set_all.npy', allow_pickle=True).item()

    return training_dis_set, training_drug_set, training_set_count,testing_dis_set, testing_drug_set, testing_set_count, dis_rating_set_all


def load_all(dataset_base_path,dis_num,drug_num):
    training_dis_set, training_drug_set, training_set_count, testing_dis_set, testing_drug_set, testing_set_count, drug_rating_set_all = load_datanpy(dataset_base_path)
    dis_d = readD(training_dis_set, dis_num)
    drug_d = readD(training_drug_set, drug_num)

    d_i_train = dis_d
    d_j_train = drug_d

    # 稀疏矩阵
    sparse_di_dr = readTrainSparseMatrix(training_dis_set, True, dis_d, drug_d)
    sparse_dr_di = readTrainSparseMatrix(training_drug_set, False, dis_d, drug_d)

    return d_i_train, d_j_train, sparse_di_dr, sparse_dr_di

def node_features(features_type, init_features_file):

    if features_type == 'CMap features':
        # drug
        df_drug = pd.read_excel(init_features_file, sheet_name='drug')
        drug_fea = [i[1: -1].split(', ') for i in list(df_drug['Gene_vect'])]
        drug_fea_init = []
        for dr in drug_fea:
            drug_fea_init.append(list(map(float, dr)))

        fin_drug_fea_init = []
        for drug_fea_init_list in drug_fea_init:
            one_drug_fea_init = []
            for drug_fea in drug_fea_init_list:
                one_drug_fea_init.append(drug_fea * 0.1)
            fin_drug_fea_init.append(one_drug_fea_init)
        drug_input_fea = np.array(fin_drug_fea_init)

        # disease
        df_dis = pd.read_excel(init_features_file, sheet_name='disease')
        diease_fea = [i[1: -1].split(', ') for i in list(df_dis['Gene_vect'])]
        diease_fea_init = []
        for di in diease_fea:
            diease_fea_init.append(list(map(float, di)))

        fin_diease_fea_init = []
        for diease_fea_init_list in diease_fea_init:
            one_diease_fea_init = []
            for disease_fea in diease_fea_init_list:
                one_diease_fea_init.append(disease_fea * 0.1)
            fin_diease_fea_init.append(one_diease_fea_init)
        diease_input_fea = np.array(fin_diease_fea_init)

    if features_type == 'random features':
        df_random_drug = pd.read_excel(init_features_file, sheet_name='drug')
        drug_input_fea = df_random_drug.values
        df_random_dis = pd.read_excel(init_features_file, sheet_name='disease')
        diease_input_fea = df_random_dis.values

    if features_type == 'CMap+CL features':
        CMap_CL_fea = np.load(init_features_file)
        drug_input_fea = CMap_CL_fea[:262]
        diease_input_fea = CMap_CL_fea[262:]

    if features_type == 'random+CL features':
        random_CL_fea = np.load(init_features_file)
        drug_input_fea = random_CL_fea[:262]
        diease_input_fea = random_CL_fea[262:]

    return drug_input_fea, diease_input_fea


class BPRData_rank(data.Dataset):
    def __init__(self, drugA_drugB=None,num_drug=0, is_training=None, data_set_count=0, all_rating=None):
        super(BPRData_rank, self).__init__()

        self.num_drug = num_drug
        self.drugA_drugB = drugA_drugB


        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating = all_rating
        self.set_all_drug = set(range(num_drug))

    def __len__(self):
        return len(self.drugA_drugB)

    def __getitem__(self, idx):

        drugA_drugB = self.drugA_drugB
        dis = drugA_drugB[idx][0]
        drugA = drugA_drugB[idx][1]
        drugB = drugA_drugB[idx][2]
        return dis, drugA, drugB

class BPRData_pos_neg(data.Dataset):
    def __init__(self, pos_neg=None, num_drug=0, is_training=None, all_rating=None):
        super(BPRData_pos_neg, self).__init__()

        self.num_drug = num_drug
        self.pos_neg = pos_neg
        self.is_training = is_training
        self.all_rating = all_rating
        self.set_all_drug = set(range(num_drug))

    def __len__(self):
        return len(self.pos_neg)

    def __getitem__(self, idx):
        pos_neg = self.pos_neg
        dis = pos_neg[idx][0]
        drug_i = pos_neg[idx][1]
        drug_j = pos_neg[idx][2]
        return dis, drug_i, drug_j

def test1V5_list_format():
    args = parse_args()
    df_test1V5 = pd.read_excel(args.input_train_test,sheet_name='fin_test')

    drug_positive_id = df_test1V5['drug_positive_id']
    str_drug_positive_id_list = [i[1: -1].split(', ') for i in drug_positive_id]
    drug_positive_id_list = []
    for drug_pos_id in str_drug_positive_id_list:
        drug_positive_id_list.append(list(map(int, drug_pos_id)))

    drug_negative_id = df_test1V5['drug_negative_id']
    str_drug_negative_id_list = [i[1: -1].split(', ') for i in drug_negative_id]
    drug_negative_id_list = []
    for drug_neg_id in str_drug_negative_id_list:
        drug_negative_id_list.append(list(map(int, drug_neg_id)))

    fin_df_test1V5 = pd.DataFrame()
    fin_df_test1V5['disease_id'] = [int(disease_id) for disease_id in df_test1V5['disease_id'].tolist()]
    fin_df_test1V5['drug_positive_id'] = drug_positive_id_list
    fin_df_test1V5['drug_negative_id'] = drug_negative_id_list

    disease_id_list = []
    drug_pos_id_list = []
    drug_neg_id_list = []
    for index,row in fin_df_test1V5.iterrows():
        for drug_pos in row['drug_positive_id']:
            for i in range(5):
                disease_id_list.append(row['disease_id'])
                drug_pos_id_list.append(drug_pos)

    for drug_neg in fin_df_test1V5['drug_negative_id'].tolist():
        for drugneg in drug_neg:
            drug_neg_id_list.append(drugneg)
    df_test1v5_pos_neg = pd.DataFrame()

    return fin_df_test1V5,df_test1v5_pos_neg
