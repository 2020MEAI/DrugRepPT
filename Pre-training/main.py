import argparse
import os.path as osp
import random
import nni
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian

from pGRACE.model import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset,get_rel,get_features
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank

def train():
    model.train()
    #view_learner.eval()
    optimizer.zero_grad()
    # data.edge_index:<class 'torch.Tensor'>  torch.Size([2, 10556])
    # edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    # edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2

    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]

    # data.x:<class 'torch.Tensor'> torch.Size([2708, 1433])
    # x_1 = drop_feature(data.x, drop_feature_rate_1)#3
    # x_2 = drop_feature(data.x, drop_feature_rate_2)#4
    # 节点特征：CMap特征
    # x_1 = drop_feature(fea_init, drop_feature_rate_1)#3
    # x_2 = drop_feature(fea_init, drop_feature_rate_2)#4
    # 节点特征：随机初始化特征
    x_1 = drop_feature(fea_init, drop_feature_rate_1)#3
    x_2 = drop_feature(fea_init, drop_feature_rate_2)#4
    #cora:3,3,6,3
    #CS:(1,2)(1,2)(2,3)(2,3)
    #AP:(3,4)(4,5)(1,2)(2,3)
    #Citseer(2,3)(3,4)(1,2)(1,2)(2,2)
    #CiteSeer(4,2)(3,2)
    #AC:(3,4)(1,4)(0,2)(1,3)
    #PubMed:(0,3)(1,3)(0,3)(0,2)
    #k2 = np.random.randint(0, 4)
    z1 = model(x_1, edge_index_1, [2, 2])
    z2 = model(x_2, edge_index_2, [8, 8])

    loss = model.loss(z1, z2, batch_size=64 if dataset == 'Coauthor-Phy' or dataset == 'ogbn-arxiv' else None)
    loss.backward()
    optimizer.step()

    return loss.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_file', type=str, default='../data/Pre-training/input_data.xlsx')
    parser.add_argument('--fea_type', default='CMap features',choices=['CMap features', 'random features'],type=str)
    parser.add_argument('--sim_type', default='all_right', choices=['only_drug_dis','only_drug_0.5','only_dis_0.5', 'all_0.5','all_right' ], type=str)
    parser.add_argument('--output_features_file', default='../data/Fine-tuning/',type=str)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)['Cora']

    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(args.seed)
    use_nni = args.config == 'nni'
    learning_rate = config['learning_rate']
    # num_hidden = config['num_hidden']
    # num_proj_hidden = config['num_proj_hidden']
    num_hidden = 512
    num_proj_hidden = 512
    activation = config['activation']
    base_model = config['base_model']
    num_layers = config['num_layers']
    dataset = 'Cora'
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_scheme = config['drop_scheme']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    rand_layers = config['rand_layers']


    device = torch.device(args.device)

    fea_init = get_features(args.fea_type,args.input_file).to(device)

    edge_index = get_rel(args.input_file,args.sim_type).to(device)



    adj = 0
    

    encoder = NewEncoder(fea_init.shape[1], num_hidden, get_activation(activation),
                         base_model=NewGConv, k=num_layers).to(device)

    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )   

    log = args.verbose.split(',')

    for epoch in range(1, num_epochs + 1):

        loss = train()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % 100 == 0:

            # node features
            x_1 = drop_feature(fea_init, drop_feature_rate_1)#3
            x_2 = drop_feature(fea_init, drop_feature_rate_2)#4

            edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2
            edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0] #adjacency with edge droprate 2

            z = model(fea_init, edge_index, [2, 2], final=True).detach().cpu().numpy()
            z1 = model(x_1, edge_index_1, [2, 2], final=True).detach().cpu().numpy()
            z2 = model(x_2, edge_index_2, [2, 2], final=True).detach().cpu().numpy()

            if args.fea_type == 'CMap features':
                # np.save(args.output_features_file+'node_features/CMap_CL_features/Graph_embeddingfull_fea.npy', z)
                np.save(args.output_features_file + 'node_features/CMap_CL_features/'+args.sim_type+'_Graph_embeddingfull_fea.npy', z)
            elif args.fea_type == 'random features':
                np.save(args.output_features_file + 'node_features/random+CL features/Graph_embeddingfull_fea.npy', z)




