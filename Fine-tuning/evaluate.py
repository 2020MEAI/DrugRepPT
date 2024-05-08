# -- coding:UTF-8
import numpy as np
import torch 
import time
import pdb
import math

def LTR_loss(user,item_i,item_j,dis, drugA, drugB,prediction_i, prediction_j,prediction_A, prediction_B,beta):
    loss_rank = -((prediction_A - prediction_B).sigmoid().log().mean())
    #print('loss_rank:',loss_rank)
    loss_pos_neg = -((prediction_i - prediction_j).sigmoid().log().mean())
    #print('loss_pos_neg:',loss_pos_neg)
    l2_pos_neg_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
    loss_pos_neg_regulization = l2_pos_neg_regulization.mean()
    #print('loss_pos_neg_regulization:',loss_pos_neg_regulization)
    l2_rank_regulization = 0.01 * (dis ** 2 + drugA ** 2 + drugB ** 2).sum(dim=-1)
    loss_rank_regulization = l2_rank_regulization.mean()
    #print('loss_rank_regulization:',loss_rank_regulization)
    # loss = 0.001 * loss_rank + 10 * loss_pos_neg +loss_pos_neg_regulization + loss_rank_regulization
    # loss1 = loss_pos_neg +  0 * loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    # print('L1:',loss1)
    loss = (loss_pos_neg+loss_rank)/(loss_pos_neg+loss_rank+beta) * loss_pos_neg + beta/(loss_pos_neg+loss_rank+beta) *loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    # loss = (loss_pos_neg + loss_rank / (loss_pos_neg + loss_rank + beta)) * math.log(loss_pos_neg) + (beta / (loss_pos_neg + loss_rank + beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization

    # loss = (loss_pos_neg / (loss_pos_neg + beta)) * loss_pos_neg + (beta / (loss_pos_neg+ beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization
    # loss = (loss_rank / (loss_rank + beta)) * loss_pos_neg + (beta / (loss_rank + beta)) * loss_rank + loss_pos_neg_regulization + loss_rank_regulization

    return loss, loss_rank, loss_pos_neg

def test_LTR_loss(user,item_i,item_j,prediction_i, prediction_j):
    #loss_rank = -((prediction_A - prediction_B).sigmoid().log().mean())
    loss_pos_neg = -((prediction_i - prediction_j).sigmoid().log().mean())
    l2_pos_neg_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
    loss_pos_neg_regulization = l2_pos_neg_regulization.mean()

    loss = loss_pos_neg +loss_pos_neg_regulization
    return loss

def rank_metrics_loss(model, testing_loader, batch_size,beta):
    start_time = time.time()
    test_loss_pos_neg_list = []
    test_loss_rank_list = []
    loss_sum = []
    #for user, item_i, item_j in testing_loader:
    for step, ((user, item_i, item_j), (dis, drugA, drugB)) in enumerate(testing_loader):
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        dis = dis.cuda()
        drugA = drugA.cuda()
        drugB = drugB.cuda()

        # prediction_i, prediction_j, loss, loss2, gcn_users_embedding, gcn_items_embedding,str_user,str_item = model(user, item_i, item_j)
        # prediction_i, prediction_j = model(user, item_i, item_j)
        # prediction_A, prediction_B = model(dis, drugA, drugB)
        prediction_i, prediction_j,user, item_i, item_j = model(user, item_i, item_j)
        prediction_A, prediction_B,dis, drugA, drugB = model(dis, drugA, drugB)
        loss, loss_rank, loss_pos_neg = LTR_loss(user,item_i,item_j,dis, drugA, drugB,prediction_i, prediction_j, prediction_A, prediction_B,beta)
        #loss = LTR_loss(user, item_i, item_j, prediction_i, prediction_j)

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
    for user, item_i, item_j in test_val_loader_loss:
        # user = user.cuda()
        # item_i = item_i.cuda()
        # item_j = item_j.cuda()
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()
     
        
        #prediction_i, prediction_j, loss, loss2, gcn_users_embedding, gcn_items_embedding,str_user,str_item = model(user, item_i, item_j) 
        prediction_i, prediction_j, loss, loss2 = model(user, item_i, item_j) 
        loss_sum.append(loss.item())  
        loss_sum2.append(loss2.item())

        # if np.isnan(loss2.item()).any():
        #     pdb.set_trace()
    # pdb.set_trace()
    elapsed_time = time.time() - start_time
    test_val_loss1=round(np.mean(loss_sum),4)
    test_val_loss=round(np.mean(loss_sum2),4)#round(np.mean(loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_val_loss=' val loss:'+str(test_val_loss)#+' time:'+str(round(elapsed_time,3))+' s'
    # print(round(elapsed_time,3))
    # print(test_val_loss1,test_val_loss)
    return test_val_loss

 

def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break

    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg  
    # hr_t,ndcg_t,index_end_i,indices_sort_top
    # pdb.set_trace() 
    return hr_t,ndcg_t
 
def HR_MRR(u_i_truth_item_list, u_i_pred_item_list):
    h1_i = 0
    h3_i = 0
    h10_i = 0
    mr_i = 0
    for u_i_truth_item in u_i_truth_item_list:
        mr_i += 1 / (int(u_i_pred_item_list.index(u_i_truth_item)) + 1)  # 返回u_i_truth_item在u_i_pred_itme_list中的位置index
        if u_i_truth_item in u_i_pred_item_list[0:1]:
            h1_i += 1
        if u_i_truth_item in u_i_pred_item_list[:3]:
            h3_i += 1
        if u_i_truth_item in u_i_pred_item_list[:10]:
            h10_i += 1
    u_i_truth_item_num = len(u_i_truth_item_list)
    # return h10_i / u_i_truth_item_num, h3_i / u_i_truth_item_num, h1_i / u_i_truth_item_num, mr_i / u_i_truth_item_num
    return h10_i, h3_i, h1_i, mr_i,u_i_truth_item_num

def HR_MRR_n(u_i_truth_item_list, u_i_pred_item_list):
    h1_i = 0
    h2_i = 0
    h3_i = 0
    h4_i = 0
    h5_i = 0
    h6_i = 0
    h7_i = 0
    h8_i = 0
    h9_i = 0
    h10_i = 0
    mr_i = 0
    for u_i_truth_item in u_i_truth_item_list:
        mr_i += 1 / (int(u_i_pred_item_list.index(u_i_truth_item)) + 1)  # 返回u_i_truth_item在u_i_pred_itme_list中的位置index
        if u_i_truth_item in u_i_pred_item_list[0:1]:
            h1_i += 1
        if u_i_truth_item in u_i_pred_item_list[:2]:
            h2_i += 1
        if u_i_truth_item in u_i_pred_item_list[:3]:
            h3_i += 1
        if u_i_truth_item in u_i_pred_item_list[:4]:
            h4_i += 1
        if u_i_truth_item in u_i_pred_item_list[:5]:
            h5_i += 1
        if u_i_truth_item in u_i_pred_item_list[:6]:
            h6_i += 1
        if u_i_truth_item in u_i_pred_item_list[:7]:
            h7_i += 1
        if u_i_truth_item in u_i_pred_item_list[:8]:
            h8_i += 1
        if u_i_truth_item in u_i_pred_item_list[:9]:
            h9_i += 1
        if u_i_truth_item in u_i_pred_item_list[:10]:
            h10_i += 1
    u_i_truth_item_num = len(u_i_truth_item_list)
    # return h10_i / u_i_truth_item_num, h3_i / u_i_truth_item_num, h1_i / u_i_truth_item_num, mr_i / u_i_truth_item_num
    return h10_i, h3_i, h1_i, mr_i,u_i_truth_item_num

def NDCG(u_i_truth_item_list, u_i_pred_item_list):
    HITS_i = 0
    DCG_i = 0
    IDCG_i = 0
    NDCG_i = 0
    for j in range(len(u_i_pred_item_list)):
        if u_i_pred_item_list[j] in u_i_truth_item_list:
            HITS_i += 1
            IDCG_i += 1.0 / math.log2((HITS_i) + 1)
            DCG_i += 1.0 / math.log2((u_i_pred_item_list.index(u_i_pred_item_list[j]) + 1) + 1)
    if HITS_i > 0:
        NDCG_i = DCG_i / IDCG_i
    return NDCG_i
    
  
def metrics(model, test_val_loader, top_k, num_negative_test_val, batch_size):
    HR, NDCG = [], [] 
    test_loss_sum=[]
    # pdb.set_trace()  
 
    test_start_time = time.time()
    for user, item_i, item_j in test_val_loader:  
        # start_time = time.time()
        # pdb.set_trace()
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j #index to split
        prediction_i, prediction_j,loss_test,loss2_test = model(user, item_i, torch.cuda.LongTensor([0]))


        # prediction_i, prediction_j,loss_test,loss2_test = model(user, item_i, torch.LongTensor([0]))

        test_loss_sum.append(loss2_test.item())  
        # pdb.set_trace()   
        elapsed_time = time.time() - test_start_time
        print('time:'+str(round(elapsed_time,2)))
        courrent_index=0
        courrent_user_index=0
        for len_i,len_j in item_j:
            index_end_i=(len_i-len_j).item()  
            #pre_error=(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)])#.sum() 
            #loss_test=nn.MSELoss((pre_error).sum())#-(prediction_i[0][courrent_index:(courrent_index+index_end_i)]- prediction_i[0][(courrent_index+index_end_i):(courrent_index+index_end_j)]).sigmoid().log()#.sum()   
            _, indices = torch.topk(prediction_i[0][courrent_index:(courrent_index+len_i)], top_k)   
            hr_t,ndcg_t=hr_ndcg(indices.tolist(),index_end_i,top_k)  
            # print(hr_t,ndcg_t,indices,index_end_i)
            # pdb.set_trace()
            HR.append(hr_t)
            NDCG.append(ndcg_t) 

            courrent_index+=len_i 
            courrent_user_index+=1 

 
    test_loss=round(np.mean(test_loss_sum[:-1]),4)  
 
    return test_loss,round(np.mean(HR),4) , round(np.mean(NDCG),4) 



