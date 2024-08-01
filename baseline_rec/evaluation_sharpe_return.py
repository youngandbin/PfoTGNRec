import yaml
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
SEED = 2022

from logging import getLogger
from recbole.config import Config
from recbole.utils.utils import init_seed
from recbole.utils.logger import init_logger
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils.utils import get_model
from recbole.utils.utils import get_trainer

# from recbole.trainer.hyper_tuning import HyperTuning
from recbole.quick_start.quick_start import objective_function
from recbole.data.dataset import Dataset

from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk

import argparse
import pickle
import time
import json

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='RecBole')
    parser.add_argument('--model', type=str, default='BPR', help='model name')
    parser.add_argument('--k', type=int, default=10, help='topk')
    
    return parser.parse_args()

def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))

def ndcg_at_k(recommendations, test_items, k):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_items)))])
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg

def main(params):
    # 1번 gpu에서 돌아가게 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # params = arg_parse
    dir = '/workspace/best_saved'
    best_saved = sorted(os.listdir(dir))
    # print(best_saved)
    model_name = params.model
    k = params.k
    # print(best_saved)
    best_saved_model = [x for x in best_saved if model_name in x][0]
    
    print('start!')
    print('Model:', model_name, ' k:', k, ' best_saved_model:', best_saved_model)
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=dir + '/' + best_saved_model)
    
    print('Load data and model is done!') 
    
    # dataset들 불러오기
    # transaction data
    with open('/workspace/dataset/ml_transaction.json', 'r') as f:
        ml_transaction = json.load(f)
    whole_dataset = pd.DataFrame(ml_transaction)
    

    # 시간 mapping dataset
    with open('/workspace/dataset/map_ts_id.pkl', 'rb') as f:
        map_ts_id = pickle.load(f)
    map_id_ts = {v: k for k, v in map_ts_id.items()}

    # item mapping dataset
    with open('/workspace/dataset/map_item_id.pkl', 'rb') as f:
        new_map_id_item = pickle.load(f)
    # map_id_item의 key는 int인데, 이를 str로 바꿔주고 길이가 6이 되도록 앞에 0을 채워줌
    map_id_item = {str(k).zfill(6): v for k, v in new_map_id_item.items()}
    
    whole_dataset['ts_real'] = whole_dataset['ts'].apply(lambda x: map_id_ts[x])
    whole_dataset['ts_real'] = pd.to_datetime(whole_dataset['ts_real'], format='%Y%m%d')
    whole_dataset['u'] = whole_dataset['u'].astype(str)
    whole_dataset['i'] = whole_dataset['i'].astype(str)
    
    
    start_time = pd.to_datetime('20221026', format='%Y%m%d') #20230327
    finish_time = pd.to_datetime('20230703', format='%Y%m%d')
    whole_dataset = whole_dataset[(whole_dataset['ts_real'] >= start_time) & (whole_dataset['ts_real'] <= finish_time)]
    test_user = whole_dataset['u'].unique().tolist()
    uid_series = dataset.token2id(dataset.uid_field, test_user)
    
    print('Data preprocessing is done!')
    
    with open('/workspace/dataset/period_all/time_feature_past.pkl', 'rb') as f:
        time_feature = pickle.load(f)

    change_portfolio = []
    original_return = []
    change_return = []
    original_sharpe = []
    change_sharpe = []
    recalls_1 = []
    recalls_3 = []
    recalls_5 = []
    recalls_10 = []
    recalls_20 = []
    ndcgs_1 = []
    ndcgs_3 = []
    ndcgs_5 = []
    ndcgs_10 = []
    ndcgs_20 = []
    new_map_id_item = {v: k for k, v in map_id_item.items()}
    
    test_recall_1 = 0.0
    test_recall_3 = 0.0
    test_recall_5 = 0.0
    test_recall_10 = 0.0
    test_recall_20 = 0.0
    test_ndcg_1 = 0.0
    test_ndcg_3 = 0.0
    test_ndcg_5 = 0.0
    test_ndcg_10 = 0.0
    test_ndcg_20 = 0.0
    num_inters = 0

    for i, row in tqdm(whole_dataset.iterrows(), total=whole_dataset.shape[0]):
        user_id = row['u']
        uid_series = dataset.token2id(dataset.uid_field, user_id)
        # topk_score, topk_iid_list_1 = full_sort_topk([uid_series], model, test_data, k=1, device=config['device'])
        # topk_score, topk_iid_list = full_sort_topk([uid_series], model, test_data, k=3, device=config['device'])  
        # topk_score, topk_iid_list_5 = full_sort_topk([uid_series], model, test_data, k=5, device=config['device'])
        # tok_score, topk_iid_list_10 = full_sort_topk([uid_series], model, test_data, k=10, device=config['device'])
        # topk_score, topk_iid_list_20 = full_sort_topk([uid_series], model, test_data, k=20, device=config['device'])
        
        # external_item_list_1 = dataset.id2token(dataset.iid_field, topk_iid_list_1.cpu()) # item idx
        # external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()) # item idx
        # external_item_list_5 = dataset.id2token(dataset.iid_field, topk_iid_list_5.cpu()) # item idx
        # external_item_list_10 = dataset.id2token(dataset.iid_field, topk_iid_list_10.cpu()) # item idx
        # external_item_list_20 = dataset.id2token(dataset.iid_field, topk_iid_list_20.cpu()) # item idx
        
        # time_feature[int(ts)]에서 각 주식의 return과 sharpe를 구해준다. 이때 time_feature[int(ts)]는 key는 주식코드이고 value는 이전 30일간의 종가 값이다.
        time_feature_returns = {}
        time_feature_daily_return = {}
        time_feature_sharpes = {}
        
        for stock in time_feature[int(row['ts'])].keys():
            time_feature_returns[stock] = np.log(time_feature[int(row['ts'])][stock][1:] / time_feature[int(row['ts'])][stock][:-1])
            time_feature_daily_return[stock] = np.mean(time_feature_returns[stock])
            time_feature_sharpes[stock] = (time_feature_daily_return[stock]*251) / (np.std(time_feature_returns[stock])*np.sqrt(251))
        
        
        # 각 time_feature_returns와 time_feature_sharpes의 value를 기준으로 높은 순서대로 정렬하고 그 key를 추출한다.
        time_feature_returns = sorted(time_feature_returns.items(), key=lambda x: x[1], reverse=True)
        time_feature_sharpes = sorted(time_feature_sharpes.items(), key=lambda x: x[1], reverse=True) 
        
        # calculate original portfolio return and sharpe
        port_feature = []
        ts = row['ts']
        count = 0
        for p in row['portfolio']:
            p = str(p).zfill(6) # p는 int가 들어간 list이지만 이를 str로 바꿔주고, 항상 길이가 6이 되도록 0을 앞에 채워줌
            # p = map_id_item[p] # 이때의 p는 주식코드이므로 이를 map_id_item dictionary를 사용해서 id로 바꿔준다
            port_feature.append(time_feature[int(ts)][p])
            

        port_feature = np.array(port_feature)
        daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1])
        daily_return = np.mean(daily_return, axis=0)
        sharpe = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
        port_return = np.mean(daily_return)*251
        original_return.append(port_return)
        original_sharpe.append(sharpe)
        
        # calculate change portfolio return and sharpe
        # recommend_list는 external_item_list[0] 내 값들을 int로 바꾼 후 13626를 빼주고, 그 후에 map_id_item의 key와 value를 바꾼 값에서 매핑한 값이다.
        port_feature = port_feature.tolist()
        if len(list(topk_iid_list.cpu())) != 0:
            # calculate the recall and ndcg
            num_inters += 1
            user_recall_1 = recall_at_k(external_item_list_1[0], [str(row['i'])], 1)
            user_recall_3 = recall_at_k(external_item_list[0], [str(row['i'])], 3)
            user_recall_5 = recall_at_k(external_item_list_5[0], [str(row['i'])], 5)
            user_recall_10 = recall_at_k(external_item_list_10[0], [str(row['i'])], 10)
            user_recall_20 = recall_at_k(external_item_list_20[0], [str(row['i'])], 20)
            recalls_1.append(user_recall_1)
            recalls_3.append(user_recall_3)
            recalls_5.append(user_recall_5)
            recalls_10.append(user_recall_10)
            recalls_20.append(user_recall_20)
            test_recall_1 += user_recall_1
            test_recall_3 += user_recall_3
            test_recall_5 += user_recall_5
            test_recall_10 += user_recall_10
            test_recall_20 += user_recall_20
            
            user_ndcg_1 = ndcg_at_k(external_item_list_1[0], [str(row['i'])], 1)
            user_ndcg_3 = ndcg_at_k(external_item_list[0], [str(row['i'])], 3)
            user_ndcg_5 = ndcg_at_k(external_item_list_5[0], [str(row['i'])], 5)
            user_ndcg_10 = ndcg_at_k(external_item_list_10[0], [str(row['i'])], 10)
            user_ndcg_20 = ndcg_at_k(external_item_list_20[0], [str(row['i'])], 20)
            ndcgs_1.append(user_ndcg_1)
            ndcgs_3.append(user_ndcg_3)
            ndcgs_5.append(user_ndcg_5)
            ndcgs_10.append(user_ndcg_10)
            ndcgs_20.append(user_ndcg_20)
            test_ndcg_1 += user_ndcg_1
            test_ndcg_3 += user_ndcg_3
            test_ndcg_5 += user_ndcg_5
            test_ndcg_10 += user_ndcg_10
            test_ndcg_20 += user_ndcg_20
            
            # print('---------------------------------------')
            # print(external_item_list[0],[str(row['i'])])
            # print(external_item_list_5[0], [str(row['i'])])
            # print('Recall@3:', user_recall_3, 'Recall@5:', user_recall_5)
            # print('NDCG@3:', user_ndcg_3, 'NDCG@5:', user_ndcg_5)
            
            change_portfolio_list = [int(i)-13462 for i in external_item_list[0]] # 0부터 시작
            change_portfolio_list = [new_map_id_item[i] for i in change_portfolio_list] # 주식코드
            change_portfolio.append(change_portfolio_list)
            for n in change_portfolio_list:
                port_feature.append(time_feature[int(ts)][n])
            port_feature = np.array(port_feature)
            daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1])
            daily_return = np.mean(daily_return, axis=0)
            sharpe = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
            port_return = (np.mean(daily_return)*251)
            change_return.append(port_return)
            change_sharpe.append(sharpe)
        else:
            change_portfolio.append([])
            change_return.append(np.nan)
            change_sharpe.append(np.nan)
            recalls_1.append(np.nan)
            recalls_3.append(np.nan)
            recalls_5.append(np.nan)
            recalls_10.append(np.nan)
            recalls_20.append(np.nan)
            ndcgs_1.append(np.nan)
            ndcgs_3.append(np.nan)
            ndcgs_5.append(np.nan)
            ndcgs_10.append(np.nan)
            ndcgs_20.append(np.nan)

    whole_dataset['change_portfolio'] = change_portfolio
    whole_dataset['original_return'] = original_return
    whole_dataset['change_return'] = change_return
    whole_dataset['original_sharpe'] = original_sharpe
    whole_dataset['change_sharpe'] = change_sharpe
    whole_dataset['recall_1'] = recalls_1
    whole_dataset['recall_3'] = recalls_3
    whole_dataset['recall_5'] = recalls_5
    whole_dataset['recall_10'] = recalls_10
    whole_dataset['recall_20'] = recalls_20
    whole_dataset['ndcg_1'] = ndcgs_1
    whole_dataset['ndcg_3'] = ndcgs_3
    whole_dataset['ndcg_5'] = ndcgs_5
    whole_dataset['ndcg_10'] = ndcgs_10
    whole_dataset['ndcg_20'] = ndcgs_20

    
    print('Calculating return and sharpe is done!')
    print('Recall@3:', test_recall_3/num_inters, 'Recall@5:', test_recall_5/num_inters, 'Recall@10:', test_recall_10/num_inters, 'Recall@20:', test_recall_20/num_inters)
    print('NDCG@3:', test_ndcg_3/num_inters, 'NDCG@5:', test_ndcg_5/num_inters, 'NDCG@10:', test_ndcg_10/num_inters, 'NDCG@20:', test_ndcg_20/num_inters)
    
    
    
    transaction = whole_dataset
    # transaction = transaction[['u', 'i', 'ts', 'ts_real', 'porfolio', 'change_portfolio', 'original_sharpe', 'change_sharpe', 'Return_change', 'Sharpe_change', 'Return_percent', 'Sharpe_percent']]
    
    transaction['Return_change'] = transaction['original_return']-transaction['change_return']
    transaction['Sharpe_change'] = transaction['original_sharpe']-transaction['change_sharpe']
    transaction['Return_percent'] = transaction['original_return']<transaction['change_return']
    transaction['Sharpe_percent'] = transaction['original_sharpe']<transaction['change_sharpe']
    
    print('Making dataframe is done!')
    
    # save the dataframe
    transaction_for_save = transaction[['u', 'i', 'ts', 'ts_real', 'portfolio', 'change_portfolio', 'original_sharpe', 'change_sharpe', 'Return_change', 'Sharpe_change', 'Return_percent', 'Sharpe_percent',
                                        'recall_1', 'recall_3', 'recall_5', 'recall_10', 'recall_20', 
                                        'ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10', 'ndcg_20']]
    transaction_for_save.to_csv('/workspace/result_insample/'+model_name+'_'+str(k)+'_transaction.csv', index=False)
    print(model_name+'_'+str(k)+'_transaction.csv',': Saving csv is done!')
        
        
if __name__ == '__main__':
    params = arg_parse()
    main(params)