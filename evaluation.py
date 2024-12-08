import math
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder

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

def return_sharpe_at_k(ts, port_feature, return_, sharpe, time_feature, pos_neg_item, k):

    # 2. new portfolio's performance
    # append top k items to port_feature
    items_to_append = pos_neg_item[:k]                                            # [55, 859, 1021]
    features_to_append = np.array([time_feature[ts][c] for c in items_to_append]) # shape: (k, 30)

    port_feature_new = np.concatenate([port_feature, features_to_append], axis=0)       # shape: (n_stocks+k, 30)
    daily_log_returns_new = np.log(port_feature_new[:, 1:] / port_feature_new[:, :-1])  # shape: (n_stocks+k, n_days-1) # price -> log return
    daily_return_new = np.mean(daily_log_returns_new, axis=0)                           # shape: (n_days-1,)
    return_new = np.mean(daily_return_new)*251
    sharpe_new = (np.mean(daily_return_new)*251) / (np.std(daily_return_new)*np.sqrt(251))

    return return_new - return_, sharpe_new - sharpe #, daily_return_new
   

def eval_recommendation(tgn, data, full_data, batch_size, n_neighbors, upper_u, period, is_test_run, EVAL): 

    time_feature_past = pickle.load(open(f'data/period_{period}/time_feature_past_{period}.pkl', 'rb'))      # Dictionary containing historical daily prices for all stocks for each timestamp (ts).
    time_feature_future = pickle.load(open(f'data/period_{period}/time_feature_future_{period}.pkl', 'rb'))  # Dictionary containing future daily prices for all stocks for each time series (ts)
    map_item_id = pickle.load(open(f'data/period_{period}/map_item_id.pkl', 'rb'))  # Used to convert stock codes in a user portfolio to item idx.

    with torch.no_grad():
        tgn = tgn.eval()
        # While usually the test batch size is as big as it fits in memory, 
        # here we keep it the same size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        """
        batch iteraction
        """
        recalls, ndcgs = [], []
        returns, returns_new, sharpes, sharpes_new = [], [], [], []
        returns_, returns_new_, sharpes_, sharpes_new_ = [], [], [], []
        daily_returns, daily_returns_new = [], []
        daily_returns_, daily_returns_new_ = [], []
          
        for batch in tqdm(range(num_test_batch), desc=f"Progress: Eval Batch"):

          s_idx = batch * TEST_BATCH_SIZE
          e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

          if e_idx == num_test_instance:
            continue

          # test run
          if is_test_run:
            if batch == 2:
              break

          # extract batch data: <class 'numpy.ndarray'>
          sources_batch = data.sources[s_idx:e_idx]           # (BATCH_SIZE,)
          destinations_batch = data.destinations[s_idx:e_idx] # (BATCH_SIZE,) # item idx
          timestamps_batch = data.timestamps[s_idx:e_idx]     # (BATCH_SIZE,)
          edge_idxs_batch = data.edge_idxs[s_idx: e_idx]      # (BATCH_SIZE,)
          portfolios_batch = data.portfolios[s_idx: e_idx]    # (BATCH_SIZE,) # stock code (6 digits)

          # 240601 
          destinations = full_data.destinations
          N_ITEMS = len(np.unique(destinations))

          # negative sampling 
          train_rand_sampler = RandEdgeSampler(sources_batch, destinations, portfolios_batch, upper_u, map_item_id, seed=2024)
          negatives_batch = train_rand_sampler.sample(size=N_ITEMS)  # (BATCH_SIZE, size) # item idx

          """
          node embedding 
          """
          source_embedding, destination_embedding, negative_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                                                                        destinations_batch,
                                                                                                        negatives_batch.flatten(), # (BATCH_SIZE * size,)
                                                                                                        timestamps_batch,
                                                                                                        edge_idxs_batch,
                                                                                                        n_neighbors)
          """
          score 
          """

          bsbs = source_embedding.shape[0] 

          # reshape source and destination to (bs, 1, emb_dim) 
          source_embedding = source_embedding.view(bsbs, 1, -1)
          destination_embedding = destination_embedding.view(bsbs, 1, -1)

          # reshape negative to (bs, size, emb_dim)
          negative_embedding = negative_embedding.view(bsbs, N_ITEMS, -1)

          # scores ( <class 'numpy.ndarray'> )
          pos_scores = torch.sum(source_embedding * destination_embedding, dim=2).cpu().numpy() # (bs, 1)
          neg_scores = torch.sum(source_embedding * negative_embedding, dim=2).cpu().numpy()    # (bs, size)
          
          """
          interaction loop
          """

          # convert item idx to stock codes 
          destinations_batch = destinations_batch - (upper_u + 1)                                           # item idx -> idx starting from 0
          destinations_batch = np.vectorize({v: k for k, v in map_item_id.items()}.get)(destinations_batch) # idx starting from 0 -> stock codes (6 digits)
          negatives_batch = negatives_batch - (upper_u + 1)                                                 # item idx -> idx starting from 0
          negatives_batch = np.vectorize({v: k for k, v in map_item_id.items()}.get)(negatives_batch)       # idx starting from 0 -> stock codes (6 digits)

          for i in range(bsbs):

            """
            recommendation evaluation
            """

            # calculate rankings based on scores 
            pos_score = pos_scores[i] # pos score for 1 item                            <class 'numpy.ndarray'>  (1,)
            neg_score = neg_scores[i] # neg score for a batch of items with size 'size' <class 'numpy.ndarray'>  (100,)

            scores = np.concatenate((pos_score, neg_score))  # [0.09, 0.88, 0.22, 0.15]
            ranking = np.argsort(scores)[::-1]               # [1, 2, 3, 0]

            # recall, ndcg 구하기
            pos_ranking = [0] # In the ranking, the position of the positive item is always 0 # If there are multiple positive items, pos_ranking = [0, 1, 2, ..., len_pos_item-1]]
            topk = [1, 3, 5]
            recall = [recall_at_k(ranking, pos_ranking, top) for top in topk]
            ndcg = [ndcg_at_k(ranking, pos_ranking, top) for top in topk]

            """
            investment evaluation
            """

            ts = timestamps_batch[i]          # ts:  1001604
            ts = str(ts)[:8]                  # # Use only up to the DAY
            portfolio = portfolios_batch[i]   # portfolio:  [427, 55, 859, 1021, 863] 

            # 240518
            if '' in portfolio:
              return_, sharpe, return__, sharpe_ = 0, 0, 0, 0
              # port_feature is empty and it should be concatenated with another np array with shape of (k, 30)
              port_feature = np.empty((0, int(period)))
              port_feature_ = np.empty((0, int(period)))
              
            else:
              # original portfolio performance (in sample)
              port_feature = np.array([time_feature_past[ts][p] for p in portfolio])   # shape: (n_stocks, n_days)
              daily_log_returns = np.log(port_feature[:, 1:] / port_feature[:, :-1])  # shape: (n_stocks, n_days-1)
              daily_return = np.mean(daily_log_returns, axis=0)                       # shape: (n_days-1,)
              return_ = (np.mean(daily_return)*251)
              sharpe = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
              # daily_returns.append(daily_return)

              # original portfolio performance (out sample)
              port_feature_ = np.array([time_feature_future[ts][p] for p in portfolio])   # shape: (n_stocks, n_days)
              daily_log_returns_ = np.log(port_feature_[:, 1:] / port_feature_[:, :-1])   # shape: (n_stocks, n_days-1)
              daily_return_ = np.mean(daily_log_returns_, axis=0)                         # shape: (n_days-1,)
              return__ = (np.mean(daily_return_)*251)
              sharpe_ = (np.mean(daily_return_)*251) / (np.std(daily_return_)*np.sqrt(251))
              # daily_returns_.append(daily_return_)

            # Sort the pos and neg items by ranking 
            pos_item = destinations_batch[i]      # pos item for 1 item                               <class 'numpy.int64'>
            neg_item = negatives_batch[i]         # neg item for a batch of items with size 'size'    <class 'numpy.ndarray'>
            pos_neg_item = np.concatenate(([pos_item], neg_item), axis=0)   # [427, 55, 859, 1021]
            pos_neg_item = pos_neg_item[ranking]                            # [55, 859, 1021, 427]

            # in sample
            return_diff_1, sharpe_diff_1 = return_sharpe_at_k(ts, port_feature, return_, sharpe, time_feature_past, pos_neg_item, 1)
            return_diff_3, sharpe_diff_3 = return_sharpe_at_k(ts, port_feature, return_, sharpe, time_feature_past, pos_neg_item, 3)
            return_diff_5, sharpe_diff_5 = return_sharpe_at_k(ts, port_feature, return_, sharpe, time_feature_past, pos_neg_item, 5)
            # out sample
            return_diff_1_, sharpe_diff_1_ = return_sharpe_at_k(ts, port_feature_, return__, sharpe_, time_feature_future, pos_neg_item, 1)
            return_diff_3_, sharpe_diff_3_ = return_sharpe_at_k(ts, port_feature_, return__, sharpe_, time_feature_future, pos_neg_item, 3)
            return_diff_5_, sharpe_diff_5_ = return_sharpe_at_k(ts, port_feature_, return__, sharpe_, time_feature_future, pos_neg_item, 5)

            # # save daily_returns
            # daily_returns_new.append([daily_return_new_1, daily_return_new_3, daily_return_new_5])
            # daily_returns_new_.append([daily_return_new_1_, daily_return_new_3_, daily_return_new_5_])
 
            """
            store results
            """
            recalls.append(recall)            # e.g., [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], ...]
            ndcgs.append(ndcg)
            # in sample
            returns.append([return_diff_1, return_diff_3, return_diff_5])
            sharpes.append([sharpe_diff_1, sharpe_diff_3, sharpe_diff_5])
            # out sample
            returns_.append([return_diff_1_, return_diff_3_, return_diff_5_])
            sharpes_.append([sharpe_diff_1_, sharpe_diff_3_, sharpe_diff_5_])
        
        # recommendation performances
        recall_avgs = [np.mean([recalls[i][top] for i in range(len(recalls))]) for top in range(len(topk))] # top1, top3, top5
        ndcg_avgs = [np.mean([ndcgs[i][top] for i in range(len(ndcgs))]) for top in range(len(topk))] # top1, top3, top5  

        # investment performance (in sample)
        return_avgs = [np.mean([returns[i][top] for i in range(len(returns))]) for top in range(len(topk))] # top1, top3, top5
        return_percents = [len([i for i in returns if i[top] > 0]) / len(returns) for top in range(len(topk))] # top1, top3, top5
        sharpe_avgs = [np.mean([sharpes[i][top] for i in range(len(sharpes))]) for top in range(len(topk))] # top1, top3, top5
        sharpe_percents = [len([i for i in sharpes if i[top] > 0]) / len(sharpes) for top in range(len(topk))] # top1, top3, top5

        # investment performance (out sample)
        return_avgs_ = [np.mean([returns_[i][top] for i in range(len(returns_))]) for top in range(len(topk))] # top1, top3, top5
        return_percents_ = [len([i for i in returns_ if i[top] > 0]) / len(returns_) for top in range(len(topk))] # top1, top3, top5
        sharpe_avgs_ = [np.mean([sharpes_[i][top] for i in range(len(sharpes_))]) for top in range(len(topk))] # top1, top3, top5
        sharpe_percents_ = [len([i for i in sharpes_ if i[top] > 0]) / len(sharpes_) for top in range(len(topk))] # top1, top3, top5

        # store results as dict
        eval_dict = {f'{EVAL}_recall_avg_1': recall_avgs[0],
                      f'{EVAL}_recall_avg_3': recall_avgs[1],
                      f'{EVAL}_recall_avg_5': recall_avgs[2],
                      f'{EVAL}_ndcg_avg_1': ndcg_avgs[0],
                      f'{EVAL}_ndcg_avg_3': ndcg_avgs[1],
                      f'{EVAL}_ndcg_avg_5': ndcg_avgs[2],
                      # in sample
                      f'{EVAL}_return_avg_1': return_avgs[0],
                      f'{EVAL}_return_avg_3': return_avgs[1],
                      f'{EVAL}_return_avg_5': return_avgs[2],
                      f'{EVAL}_return_percent_1': return_percents[0],
                      f'{EVAL}_return_percent_3': return_percents[1],
                      f'{EVAL}_return_percent_5': return_percents[2],
                      f'{EVAL}_sharpe_avg_1': sharpe_avgs[0],
                      f'{EVAL}_sharpe_avg_3': sharpe_avgs[1],
                      f'{EVAL}_sharpe_avg_5': sharpe_avgs[2],
                      f'{EVAL}_sharpe_percent_1': sharpe_percents[0],
                      f'{EVAL}_sharpe_percent_3': sharpe_percents[1],
                      f'{EVAL}_sharpe_percent_5': sharpe_percents[2],
                      # out sample
                      f'{EVAL}_return_avg_1_': return_avgs_[0],
                      f'{EVAL}_return_avg_3_': return_avgs_[1],
                      f'{EVAL}_return_avg_5_': return_avgs_[2],
                      f'{EVAL}_return_percent_1_': return_percents_[0],
                      f'{EVAL}_return_percent_3_': return_percents_[1],
                      f'{EVAL}_return_percent_5_': return_percents_[2],
                      f'{EVAL}_sharpe_avg_1_': sharpe_avgs_[0],
                      f'{EVAL}_sharpe_avg_3_': sharpe_avgs_[1],
                      f'{EVAL}_sharpe_avg_5_': sharpe_avgs_[2],
                      f'{EVAL}_sharpe_percent_1_': sharpe_percents_[0],
                      f'{EVAL}_sharpe_percent_3_': sharpe_percents_[1],
                      f'{EVAL}_sharpe_percent_5_': sharpe_percents_[2]
                      }
        # store_dict = {'daily_returns': daily_returns,
        #               'daily_returns_': daily_returns_,
        #               'daily_returns_new': daily_returns_new,
        #               'daily_returns_new_': daily_returns_new_,
        #               }
        
        return eval_dict #, store_dict