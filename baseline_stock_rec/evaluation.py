import numpy as np
from lib.utils import get_mean_variance, get_data, get_ret_data, load_pickle
from lib.analysis_utils import get_model, get_name
import os
import pandas as pd
from sklearn import metrics
import pickle
import sys
from tqdm import tqdm
import argparse


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
    
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

def mrr_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1 / (i + 1)
    return 0


def rec_test_analysis(total_score, holdings_data, target_year, num_rec, is_in_sample=True):
    test_data = holdings_data["test_data"]
    indptr = holdings_data["test_indptr"]
    portfolios = holdings_data["test_portfolios"]
    timestamps = holdings_data["test_timestamps"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]
    y_true = np.array(test_data[2])
    
    time_feature_past = pickle.load(open(f'data/mock/time_feature_past.pkl', 'rb'))
    time_feature_future = pickle.load(open(f'data/mock/time_feature_future.pkl', 'rb'))
    map_item_id = pickle.load(open(f'data/mock/{target_year}/map_item_id.pkl', 'rb'))
    # map_item_id의 key와 value를 바꿔준다.
    map_item_id = {v:k for k,v in map_item_id.items()}
    
    time_feature = time_feature_future # time_feature_past if is_in_sample else time_feature_future

    recalls, ndcgs, mrrs, returns, returns_new, sharpes, sharpes_new = [], [], [], [], [], [], []
    
    for inter in tqdm(range(len(test_data[0]))):
        user = test_data[0][inter]
        item = test_data[1][inter]
        
        # 1. 추천 평가
        
        user_score = total_score[user]
        true_item_score = user_score[item]
        candidate_items = list(set(range(n)) - set([item]))
        candidate_items = np.random.choice(candidate_items, 100, replace=False)
        candidate_item_score = user_score[candidate_items]
        total_item_score = np.append(true_item_score, candidate_item_score)
        ranking_item_score = np.argsort(total_item_score)[::-1]
        
        pos_ranking = [0]
        topk = [1,3,5,10,20]
        recall = [recall_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        ndcg = [ndcg_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        mrr = [mrr_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        
        # 2. 투자평가
        ts = str(timestamps[inter])
        ts = ts[:8]
        portfolio = portfolios[inter]
        
        ## 기존 portfolio performance
        ## portfolio를 돌면서 len이 6이 아니면 앞쪽에 0을 채워준다.
        portfolio = [str(p).zfill(6) for p in portfolio]
        port_feature = np.array([time_feature[ts][str(p)] for p in portfolio])  # shape: (n_stocks, 30)
        daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1]) # shape: (n_stocks, n_days-1) # price -> log return 
        daily_return = np.mean(daily_return, axis=0)                      # shape: (n_days-1,)
        sharpe_ = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
        return_ = np.mean(daily_return)*251
        
        ## 새로운 portolio performance

        ranked_item = ranking_item_score[:num_rec]
        ranked_item_code = [map_item_id[i] for i in ranked_item]
        
        features_to_append = np.array([time_feature[ts][c] for c in ranked_item_code])
        
        port_feature = np.concatenate([port_feature, features_to_append], axis=0) # shape: (n_stocks+k, 30)
        daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1])         # shape: (n_stocks+k, n_days-1) # price -> log return 
        daily_return = np.mean(daily_return, axis=0)                              # shape: (n_days-1,)
        sharpe_new = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
        return_new = np.mean(daily_return)*251
        
        ## 3. 결과 저장하기
        recalls.append(recall)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        returns.append(return_)
        returns_new.append(return_new)
        sharpes.append(sharpe_)
        sharpes_new.append(sharpe_new)
    
    eval_dict = {'recalls': recalls,
                    'ndcgs': ndcgs,
                    'mrrs': mrrs,
                    'returns': returns,
                    'returns_new': returns_new,
                    'sharpes': sharpes,
                    'sharpes_new': sharpes_new
                    }

    return eval_dict 

def rec_valid_analysis(total_score, holdings_data, target_year, num_rec, is_in_sample=True):
    test_data = holdings_data["valid_data"]
    indptr = holdings_data["valid_indptr"]
    portfolios = holdings_data["valid_portfolios"]
    timestamps = holdings_data["valid_timestamps"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]
    y_true = np.array(test_data[2])
    
    time_feature_past = pickle.load(open(f'/workspace/mvecf/data/mock/time_feature_past.pkl', 'rb'))
    time_feature_future = pickle.load(open(f'/workspace/mvecf/data/mock/time_feature_future.pkl', 'rb'))
    map_item_id = pickle.load(open(f'data/mock/{target_year}/map_item_id.pkl', 'rb'))
    # map_item_id의 key와 value를 바꿔준다.
    map_item_id = {v:k for k,v in map_item_id.items()}
    
    time_feature = time_feature_future # time_feature_past if is_in_sample else time_feature_future

    recalls, ndcgs, mrrs, returns, returns_new, sharpes, sharpes_new = [], [], [], [], [], [], []
    
    for inter in tqdm(range(len(test_data[0]))):
        user = test_data[0][inter]
        item = test_data[1][inter]
        
        # 1. 추천 평가
        
        user_score = total_score[user]
        true_item_score = user_score[item]
        candidate_items = list(set(range(n)) - set([item]))
        candidate_items = np.random.choice(candidate_items, 100, replace=False)
        candidate_item_score = user_score[candidate_items]
        total_item_score = np.append(true_item_score, candidate_item_score)
        ranking_item_score = np.argsort(total_item_score)[::-1]
        
        pos_ranking = [0]
        topk = [1,3,5,10,20]
        recall = [recall_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        ndcg = [ndcg_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        mrr = [mrr_at_k(ranking_item_score, pos_ranking, k) for k in topk]
        
        # 2. 투자평가
        ts = str(timestamps[inter])
        ts = ts[:8]
        portfolio = portfolios[inter]
        
        ## 기존 portfolio performance
        ## portfolio를 돌면서 len이 6이 아니면 앞쪽에 0을 채워준다.
        portfolio = [str(p).zfill(6) for p in portfolio]
        port_feature = np.array([time_feature[ts][str(p)] for p in portfolio])  # shape: (n_stocks, 30)
        daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1]) # shape: (n_stocks, n_days-1) # price -> log return 
        daily_return = np.mean(daily_return, axis=0)                      # shape: (n_days-1,)
        sharpe_ = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
        return_ = np.mean(daily_return)*251
        
        ## 새로운 portolio performance
        ranked_item = ranking_item_score[:num_rec]
        ranked_item_code = [map_item_id[i] for i in ranked_item]
        
        features_to_append = np.array([time_feature[ts][c] for c in ranked_item_code])
        
        port_feature = np.concatenate([port_feature, features_to_append], axis=0) # shape: (n_stocks+k, 30)
        daily_return = np.log(port_feature[:, 1:] / port_feature[:, :-1])         # shape: (n_stocks+k, n_days-1) # price -> log return 
        daily_return = np.mean(daily_return, axis=0)                              # shape: (n_days-1,)
        sharpe_new = (np.mean(daily_return)*251) / (np.std(daily_return)*np.sqrt(251))
        return_new = np.mean(daily_return)*251
        
        ## 3. 결과 저장하기
        recalls.append(recall)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        returns.append(return_)
        returns_new.append(return_new)
        sharpes.append(sharpe_)
        sharpes_new.append(sharpe_new)
    
    eval_dict = {'recalls': recalls,
                    'ndcgs': ndcgs,
                    'mrrs': mrrs,
                    'returns': returns,
                    'returns_new': returns_new,
                    'sharpes': sharpes,
                    'sharpes_new': sharpes_new
                    }

    return eval_dict 
        


def get_real_mv_user(items, ret_data_all, rebalance=False):
    ret = ret_data_all[items]

    if not rebalance:
        wealth_stock = np.cumprod(1 + ret, axis=1)
        wealth = wealth_stock.mean(axis=0)
        wealth = np.append([1], wealth)
        ret_u = (wealth[1:] / wealth[:-1]) - 1
    else:
        ret_u = np.mean(ret, axis=0)

    mu_u = ret_u.mean() * 52
    sig_u = ret_u.std() * (52 ** 0.5)

    return mu_u, sig_u


def get_sr_statistics(
        mv_list, mv_train_list, mv_insample_list, mv_insample_train_list,
):
    def get_output(mv, mv_train, is_insample=False):
        mean_list = np.array(mv).T[0]
        risk_list = np.array(mv).T[1]
        sr_list = mean_list / risk_list

        mean_train = np.array(mv_train).T[0]
        risk_train = np.array(mv_train).T[1]
        sr_train = mean_train / risk_train

        sr_diff_train = sr_list - sr_train
        mean_diff_train = mean_list - mean_train
        risk_diff_train = risk_list - risk_train

        output = {
            "delta_sr": sr_diff_train.mean(),
            "delta_mean": mean_diff_train.mean(),
            "delta_risk": risk_diff_train.mean(),
            "prob_delta_sr_positive": (sr_diff_train > 0).sum() / len(sr_list),
        }
        output = pd.DataFrame.from_dict(output, orient="index")
        if not is_insample:
            output.index = output.index + "_expost"
        return output

    output_backtest = get_output(mv_list, mv_train_list, is_insample=False)
    output_insample = get_output(mv_insample_list, mv_insample_train_list, is_insample=True)
    output = pd.concat([output_backtest, output_insample])
    return output[0].to_dict()


def mv_to_sr(mv_list):
    mv_list = np.array(mv_list).T
    sr_list = mv_list[0] / mv_list[1]
    return sr_list


def calc_sr_models(total_score, holdings_data, ret_data, mu, cov, topk=20, rebalance=False):
    train_data = holdings_data["train_data"]
    train_indptr = holdings_data["train_indptr"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]

    mv = []
    best_item = []
    recommended_all = np.zeros((m, n))
    for user in range(m):
        train_items = train_data[1][train_indptr[user]:train_indptr[user + 1]]
        if len(train_items) == 0:
            continue
        if total_score is None:
            recommended_port = train_items
        else:
            candidate_items = list(set(range(n)) - set(train_items))
            candidate_items.sort()
            candidate_items = np.array(candidate_items)
            tmp_score = total_score[user][candidate_items]
            sorted_index = tmp_score.argsort()[::-1]  # top k sorted
            sorted_items = candidate_items[sorted_index]
            # sorted_scores = tmp_score[sorted_index] # 이건 threshold 사용할 경우 필요

            best_item.append(sorted_items[0])
            recommended_port = np.append(train_items, sorted_items[:topk])
        recommended_all[user, recommended_port] = 1 / len(recommended_port)
        mean, risk = get_real_mv_user(recommended_port, ret_data, rebalance=rebalance)

        mv.append([mean, risk])
    mean_insample = np.matmul(mu, recommended_all.T)
    risk_insample = np.sqrt(np.diag(np.matmul(np.matmul(recommended_all, cov), recommended_all.T)))
    return mv, np.c_[mean_insample, risk_insample]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model analysis for financial data.')
    parser.add_argument('--data_type', type=str, default='mock', help='Type of data to use')
    parser.add_argument('--target_year', type=str, default='period_1', help='Target year for analysis')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--model_name', type=str, default='mvecf_wmf', help='Name of the model')
    parser.add_argument('--ex_post_test_years', type=int, default=5, help='Number of years for ex-post testing')
    parser.add_argument('--topk', type=int, default=1, help='Top K items to consider')
    parser.add_argument('--numbers', type=int, default=9, help='discrete number')
    
    args = parser.parse_args()

    data_type = args.data_type
    target_year = args.target_year
    save_path = args.save_path
    model_name = args.model_name
    ex_post_test_years = args.ex_post_test_years
    topk = args.topk
    numbers = args.numbers
    
    # 1번 gpu 사용
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # data_type = "mock"
    # target_year = 'period_1'

    sr_performance = {}
    rec_performance = {}
    analysis_path = os.path.join("results", "analysis.xlsx")
    index_name = ["data_type", "year", "model", "lr", "reg_param", "n_factors", "gamma", "reg_param_mv"]

    # ex_post_test_years = 5
    # topk = 20
    
    # /workspace/mvecf/results 안에 있는 모든 폴더를 가져온다.
    
    # model_list = [
    #     "wmf",
    #     # "bpr_nov",
    #     # "mvecf_reg",
    #     "mvecf_wmf",
    #     # "mvecf_lgcn",
    #     "twophase_wmf"
    # ]
    
    # load main data
    holdings_data, factor_params = get_data(data_type, target_year)
    mu, cov = get_mean_variance(factor_params)
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]

    # load return data
    ret_data = get_ret_data(data_type, target_year)
    # ret_data = ret_data[
    #     (ret_data.index.year > target_year) & (ret_data.index.year <= target_year + ex_post_test_years)
    #     ]
    ret_data = ret_data.fillna(0)
    ret_data = ret_data.T.values
    assert len(ret_data) == n

    mv_train, mv_train_insample = calc_sr_models(None, holdings_data, ret_data, mu, cov)
    sr_train = mv_to_sr(mv_train)

    for number in range(1, numbers+1):
        model_type = model_name+"_"+str(number)
        print(model_type, topk)
        
        target_year_path = os.path.join(save_path, target_year)
        dirpath = os.path.join(target_year_path, model_type)
        if "twophase" in model_type:
            total_score = load_pickle(os.path.join(dirpath, "total_score.pkl"))
        else:
            model = get_model(dirpath)
            if model is None:
                continue
            assert m == model.n_users
            assert n == model.n_items

            total_score = []
            for user in range(m):
                total_score.append(
                    model.forward(user, np.arange(model.n_items))
                )
            total_score = np.array(total_score)

            name = get_name(model, data_type, target_year, model_type, index_name)
            mv_model, mv_model_insample = calc_sr_models(
                total_score, holdings_data, ret_data, mu, cov, topk=topk)

        # sr_performance[name] = get_sr_statistics(mv_model, mv_train, mv_model_insample, mv_train_insample)
        valid_results = rec_valid_analysis(total_score, holdings_data, target_year, topk, is_in_sample=False)
        test_results = rec_test_analysis(total_score, holdings_data, target_year, topk, is_in_sample=False)
        # save results as pickle
        with open(os.path.join(dirpath, f"valid_out_results_{topk}.pkl"), "wb") as f:
            pickle.dump(valid_results, f)
        with open(os.path.join(dirpath, f"test_out_results_{topk}.pkl"), "wb") as f:
            pickle.dump(test_results, f)

