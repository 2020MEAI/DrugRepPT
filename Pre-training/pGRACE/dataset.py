import os.path as osp
import pandas as pd
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
import torch
from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')
    print (root_path)
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
#        return Coauthor(root=path, name='cs')
    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='computers')

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='photo')

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)

def get_features(fea_type,input_file):
    feature = None
    if fea_type == 'CMap features':
        df_dis_drug = pd.read_excel(input_file, sheet_name='CMap_fea')
        dis_drug_fea = [i[1: -1].split(', ') for i in list(df_dis_drug['Gene_vect'])]
        dis_drug_fea_init = []
        for dr in dis_drug_fea:
            dis_drug_fea_init.append(list(map(float, dr)))

        fin_dis_drug_fea_init = []
        for dis_drug_fea_init_list in dis_drug_fea_init:
            one_dis_drug_fea_init = []
            for dis_drug_fea in dis_drug_fea_init_list:
                one_dis_drug_fea_init.append(dis_drug_fea * 0.01)
            fin_dis_drug_fea_init.append(one_dis_drug_fea_init)
        feature = torch.Tensor(fin_dis_drug_fea_init)


    elif fea_type == 'random features':
        feature = torch.normal(0, 0.1, size=(625, 978))

    else:
        print(f"Warning: Unsupported feature type '{fea_type}'. Returning default value.")

    return feature



def get_rel(input_rel_file,sim_type):
    # dis-drug
    df_dis_drug = pd.read_excel(input_rel_file, sheet_name='dis_drug_rel')
    df_dis_drug = df_dis_drug[['dis_ID', 'drug_ID']]
    df_dis_drug.columns = ['ID1', 'ID2']
    # drug-drug
    df_drug_drug = pd.read_excel(input_rel_file, sheet_name='drug-drug')
    df_drug_drug = df_drug_drug[['ID1', 'ID2']]
    df_drug_drug_maxsim = pd.read_excel(input_rel_file, sheet_name='SIDER+DRONet_drug_drug_0.5')
    df_drug_drug_maxsim = df_drug_drug_maxsim[['ID1', 'ID2']]
    # dis-dis
    df_dis_dis = pd.read_excel(input_rel_file, sheet_name='dis-dis')
    df_dis_dis = df_dis_dis[['ID1', 'ID2']]
    df_dis_dis_maxsim = pd.read_excel(input_rel_file, sheet_name='SIDER+DRONet_dis_dis_0.5')
    df_dis_dis_maxsim = df_dis_dis_maxsim[['ID1', 'ID2']]

    if sim_type == 'only_drug_dis':
        print('only_drug_dis')
    elif sim_type == 'only_drug_0.5':
        print('only_drug_0.5')
        df_fin = pd.concat([df_dis_drug,df_drug_drug_maxsim,df_dis_dis])
    elif sim_type == 'only_dis_0.5':
        print('only_dis_0.5')
        df_fin = pd.concat([df_dis_drug,df_drug_drug,df_dis_dis_maxsim])
    elif sim_type == 'all_0.5':
        print('all_0.5')
        df_fin = pd.concat([df_dis_drug, df_drug_drug_maxsim, df_dis_dis_maxsim])
    elif sim_type == 'all_right':
        print('all_right')
        df_fin = pd.concat([df_dis_drug, df_drug_drug, df_dis_dis])

    IDt1_list = df_fin['ID1'].tolist()
    IDt2_list = df_fin['ID2'].tolist()
    edge_index_list = [IDt1_list, IDt2_list]
    edge_index = torch.tensor(edge_index_list)

    return edge_index

