# -- coding:UTF-8
import numpy as np
import torch 
import time
import pdb
import math

def LTR_loss(dis_pn,drug_i,item_j,dis, drugA, drugB,prediction_i, prediction_j,prediction_A, prediction_B,beta):
    loss_rank = -((prediction_A - prediction_B).sigmoid().log().mean())
    loss_pos_neg = -((prediction_i - prediction_j).sigmoid().log().mean())
    l2_pos_neg_regulization = 0.01 * (dis_pn ** 2 + drug_i ** 2 + item_j ** 2).sum(dim=-1)
    loss_pos_neg_regulization = l2_pos_neg_regulization.mean()
    l2_rank_regulization = 0.01 * (dis ** 2 + drugA ** 2 + drugB ** 2).sum(dim=-1)
    loss_rank_regulization = l2_rank_regulization.mean()
    #print('loss_rank_regulization:',loss_rank_regulization)
    # loss = 0.001 * loss_rank + 10 * loss_pos_neg +loss_pos_neg_regulization + loss_rank_regulization
    # loss1 = loss_pos_neg +  0 * loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    # print('L1:',loss1)
    # loss = (loss_pos_neg+loss_rank)/(loss_pos_neg+loss_rank+beta) * loss_pos_neg + beta/(loss_pos_neg+loss_rank+beta) *loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    loss = loss_rank + loss_rank_regulization
    # loss = (loss_pos_neg + loss_rank / (loss_pos_neg + loss_rank + beta)) * math.log(loss_pos_neg) + (beta / (loss_pos_neg + loss_rank + beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization

    # loss = (loss_pos_neg / (loss_pos_neg + beta)) * loss_pos_neg + (beta / (loss_pos_neg+ beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    # loss = (loss_rank / (loss_rank + beta)) * loss_pos_neg + (beta / (loss_rank + beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization

    return loss, loss_rank, loss_pos_neg

def test_LTR_loss(dis_pn,drug_i,item_j,prediction_i, prediction_j):
    loss_pos_neg = -((prediction_i - prediction_j).sigmoid().log().mean())
    l2_pos_neg_regulization = 0.01 * (dis_pn ** 2 + drug_i ** 2 + item_j ** 2).sum(dim=-1)
    loss_pos_neg_regulization = l2_pos_neg_regulization.mean()

    loss = loss_pos_neg +loss_pos_neg_regulization
    return loss

def rank_metrics_loss(model, testing_loader, batch_size,beta):
    start_time = time.time()
    test_loss_pos_neg_list = []
    test_loss_rank_list = []
    loss_sum = []

    for step, ((dis_pn, drug_i, item_j), (dis, drugA, drugB)) in enumerate(testing_loader):
        dis_pn = dis_pn.cuda()
        drug_i = drug_i.cuda()
        item_j = item_j.cuda()

        dis = dis.cuda()
        drugA = drugA.cuda()
        drugB = drugB.cuda()


        prediction_i, prediction_j,dis_pn, drug_i, item_j = model(dis_pn, drug_i, item_j)
        prediction_A, prediction_B,dis, drugA, drugB = model(dis, drugA, drugB)
        loss, loss_rank, loss_pos_neg = LTR_loss(dis_pn,drug_i,item_j,dis, drugA, drugB,prediction_i, prediction_j, prediction_A, prediction_B,beta)


        test_loss_pos_neg_list.append(loss_pos_neg.item())
        test_loss_rank_list.append(loss_rank.item())
        loss_sum.append(loss.item())

    test_loss = round(np.mean(loss_sum),4)  # round(np.mean(loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    test_loss_pos_neg = round(np.mean(test_loss_pos_neg_list),4)
    test_loss_rank = round(np.mean(test_loss_rank_list),4)
    return test_loss, test_loss_pos_neg, test_loss_rank


def metrics_loss(model, test_val_loader_loss, batch_size): 
    start_time = time.time() 
    loss_sum=[]
    loss_sum2=[]
    for dis_pn, drug_i, item_j in test_val_loader_loss:
        dis_pn = dis_pn.cuda()
        drug_i = drug_i.cuda()
        item_j = item_j.cuda()

        prediction_i, prediction_j, loss, loss2 = model(dis_pn, drug_i, item_j)
        loss_sum.append(loss.item())  
        loss_sum2.append(loss2.item())

        # if np.isnan(loss2.item()).any():
        #     pdb.set_trace()
    # pdb.set_trace()
    elapsed_time = time.time() - start_time
    test_val_loss1=round(np.mean(loss_sum),4)
    test_val_loss=round(np.mean(loss_sum2),4)#round(np.mean(loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_val_loss=' val loss:'+str(test_val_loss)#+' time:'+str(round(elapsed_time,3))+' s'

    return test_val_loss

def HR_MRR(dis_drug_truth_item_list, dis_drug_pred_item_list):
    h1_i = 0
    h3_i = 0
    h10_i = 0
    mr_i = 0
    for dis_drug_truth_item_list in dis_drug_truth_item_list:
        mr_i += 1 / (int(dis_drug_pred_item_list.index(dis_drug_truth_item_list)) + 1)
        if dis_drug_truth_item_list in dis_drug_pred_item_list[0:1]:
            h1_i += 1
        if dis_drug_truth_item_list in dis_drug_pred_item_list[:3]:
            h3_i += 1
        if dis_drug_truth_item_list in dis_drug_pred_item_list[:10]:
            h10_i += 1
    dis_drug_truth_item_list = len(dis_drug_truth_item_list)

    return h10_i, h3_i, h1_i, mr_i,dis_drug_truth_item_list



def NDCG(dis_drug_truth_item_list, dis_drug_pred_item_list):
    HITS_i = 0
    DCG_i = 0
    IDCG_i = 0
    NDCG_i = 0
    for j in range(len(dis_drug_pred_item_list)):
        if dis_drug_pred_item_list[j] in dis_drug_truth_item_list:
            HITS_i += 1
            IDCG_i += 1.0 / math.log2((HITS_i) + 1)
            DCG_i += 1.0 / math.log2((dis_drug_pred_item_list.index(dis_drug_pred_item_list[j]) + 1) + 1)
    if HITS_i > 0:
        NDCG_i = DCG_i / IDCG_i
    return NDCG_i




