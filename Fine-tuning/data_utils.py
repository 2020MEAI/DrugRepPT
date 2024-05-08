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
    user_d = []
    for i in range(num_):  # num_为用户总数or商品总数
        len_set = 1.0 / (len(set_matrix[i]) + 1)  # 1/每一行set长度+1，一共num行，算的是度分之一
        user_d.append(len_set)  # user_d即一个集合，1/每一行set长度+1
    return user_d

# user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix, is_user,u_d,i_d):
    user_items_matrix_i = []  # index
    user_items_matrix_v = []  # value
    if is_user:  # 用户对应的商品集
        d_i = u_d  # 长度=user {list:262}
        d_j = i_d  # 长度=item {list:363}
    else:  # 商品对应的用户集
        d_i = i_d  # 长度=item 这里是363
        d_j = u_d  # 长度=user 这里是262
    for i in set_matrix:
        len_set = len(set_matrix[i])  # 第i行对应药物数量（以user-item set 为例
        for j in set_matrix[i]:  # 在第一行(第一个user对应的item集合遍历)
            # print(i,j)
            user_items_matrix_i.append([i, j])  # 向矩阵内添加疾病-药物
            d_i_j = np.sqrt(d_i[i] * d_j[j])  # 疾病、药物的度相乘取根（对每对中药-基因都可以算出来一个值）
            user_items_matrix_v.append(d_i_j)  # 向矩阵内添加度

    # 正确的部分，服务器调试的时候记得改过来
    user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)  # 位置信息：疾病-药物
    user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)  # 值信息：疾病-药物（度矩阵得到）

    # user_items_matrix_i = torch.LongTensor(user_items_matrix_i)  # 中药-基因矩阵 {Tensor(30626,2)}
    # user_items_matrix_v = torch.FloatTensor(user_items_matrix_v)  # 中药-基因组的度矩阵 {Tensor(30626)}

    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)  # user_items_matrix_i转置，稀疏矩阵

def load_datanpy(dataset_base_path):
    # 训练集、测试集、验证集：中药-基因字典，基因-中药字典，基因总数

    # training_user_set: 262个疾病  training_item_set: 334个药物 training_set_count: 2111
    training_user_set, training_item_set, training_set_count = np.load(dataset_base_path + 'training_set.npy',allow_pickle=True)

    # testing_user_set: 250个疾病  testing_item_set: 203个药物 testing_set_count: 545
    testing_user_set, testing_item_set, testing_set_count = np.load(dataset_base_path + 'testing_set.npy',allow_pickle=True)

    # user_rating_set_all: 262个疾病
    user_rating_set_all = np.load(dataset_base_path + 'user_rating_set_all.npy', allow_pickle=True).item()

    return training_user_set, training_item_set, training_set_count,testing_user_set, testing_item_set, testing_set_count, user_rating_set_all


def load_all(dataset_base_path,user_num,item_num):
    training_user_set, training_item_set, training_set_count, testing_user_set, testing_item_set, testing_set_count, user_rating_set_all = load_datanpy(dataset_base_path)
    u_d = readD(training_user_set, user_num)  # 训练集中药的度分之一  {list:262}
    i_d = readD(training_item_set, item_num)  # 训练集基因的度分之一  {list:363}

    d_i_train = u_d
    d_j_train = i_d

    # 稀疏矩阵
    sparse_u_i = readTrainSparseMatrix(training_user_set, True, u_d, i_d)  # 中药对应基因，有度 {Tensor(563,2106)}
    sparse_i_u = readTrainSparseMatrix(training_item_set, False, u_d, i_d)  # 基因对应的中药，有度 {Tensor(2106,563)}

    return d_i_train, d_j_train, sparse_u_i, sparse_i_u

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
                # one_drug_fea_init.append(drug_fea)
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
    def __init__(self, drugA_drugB=None,num_item=0, is_training=None, data_set_count=0, all_rating=None):
        super(BPRData_rank, self).__init__()

        self.num_item = num_item
        self.drugA_drugB = drugA_drugB


        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating = all_rating
        self.set_all_item = set(range(num_item))  # 基因set

    def __len__(self):
        return len(self.drugA_drugB)

    def __getitem__(self, idx):

        drugA_drugB = self.drugA_drugB
        dis = drugA_drugB[idx][0]
        drugA = drugA_drugB[idx][1]
        drugB = drugA_drugB[idx][2]
        return dis, drugA, drugB

class BPRData_pos_neg(data.Dataset):
    def __init__(self, pos_neg=None, num_item=0, is_training=None, all_rating=None):
        super(BPRData_pos_neg, self).__init__()

        self.num_item = num_item
        self.pos_neg = pos_neg
        self.is_training = is_training
        self.all_rating = all_rating
        self.set_all_item = set(range(num_item))  # 基因set

    def __len__(self):
        return len(self.pos_neg)  # return self.num_ng*len(self.train_dict) 5乘1351也就是train里所有的item数*5，这个意义是生成的负样本总数=6755

    def __getitem__(self, idx):
        pos_neg = self.pos_neg
        user = pos_neg[idx][0]
        item_i = pos_neg[idx][1]
        item_j = pos_neg[idx][2]
        return user, item_i, item_j

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

    # print('disease_id:',len(disease_id_list),disease_id_list)
    # print('drug_positive_id:',len(drug_pos_id_list),drug_pos_id_list)
    # print('drug_negative_id:',len(drug_neg_id_list),drug_neg_id_list)
    df_test1v5_pos_neg = pd.DataFrame()
    # df_test1v5_pos_neg['disease_id'] = disease_id_list
    # df_test1v5_pos_neg['drug_positive_id'] = drug_pos_id_list
    # df_test1v5_pos_neg['drug_negative_id'] = drug_neg_id_list

    return fin_df_test1V5,df_test1v5_pos_neg
