import numpy as np
from lib.utils import get_data, save_pickle
from lib.analysis_utils import get_model
import os
import argparse
from tqdm import tqdm
import sys
from cvxopt import matrix, solvers

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def get_phase2_scores(sample, factors):
    num_sample = len(sample)
    beta = factors["beta"][sample]
    mu = np.matmul(factors["factor_mean"], beta.T)
    sig2_factor = factors["factor_variance"]
    sig2_eps = np.diag(factors["sig_eps_square"][sample])
    cov = np.matmul(np.matmul(beta, sig2_factor), beta.T) + sig2_eps

    gamma = 3

    Q = matrix(cov / 2 * gamma)
    p = matrix(-mu)
    G = matrix(np.concatenate([np.eye(num_sample), -np.eye(num_sample)]))
    h = matrix(np.concatenate([0.05 * np.ones(num_sample), np.zeros(num_sample)]))
    A = matrix(np.ones(num_sample), (1, num_sample))
    b = matrix(1.0)
    # A_train = np.concatenate([np.eye(num_train), np.zeros(shape=(num_train, num_sample-num_train))], axis=1)
    # A = matrix(np.concatenate([A_train, np.ones(shape=(1, num_sample))], axis=0))
    # b = matrix(np.append(np.ones(num_train)*(1/(num_train + 20)), [1]))

    sol = solvers.qp(Q, p, G, h, A, b, options={'show_progress': False})
    if sol["status"] != 'optimal':
        raise RuntimeError
    return np.array(sol['x']).flatten()


def two_phase_method(scores, holdings, factors, phase1_k=100, is_test=False):
    train_data = holdings["train_data"]
    train_indptr = holdings["train_indptr"]
    test_data = holdings["test_data"]
    test_indptr = holdings["test_indptr"]
    m = holdings["n_users"]
    n = holdings["n_items"]

    final_scores = np.zeros((m, n))
    for user in tqdm(range(m)):
        train_items = train_data[1][train_indptr[user]:train_indptr[user + 1]]
        test_items = test_data[1][test_indptr[user]:test_indptr[user + 1]]
        if is_test:
            candidate_items = test_items
        else:
            candidate_items = list(set(range(n)) - set(train_items))
        candidate_items.sort()
        candidate_items = np.array(candidate_items)
        tmp_score = scores[user][candidate_items]
        sorted_index = tmp_score.argsort()[::-1][:phase1_k]
        phase1_recs = candidate_items[sorted_index]
        recommended_port = np.append(train_items, phase1_recs)
        recommended_port = list(map(int, recommended_port))
        if len(recommended_port) == 0:
            final_scores[user][recommended_port] = 1 / len(recommended_port)
        else:
            phase2_scores = get_phase2_scores(recommended_port, factors)
            final_scores[user][recommended_port] = phase2_scores

        final_scores[user][train_items] = 0
    return final_scores


if __name__ == "__main__":
    sr_performance = {}
    rec_performance = {}
    
    # Set up argparse
    parser = argparse.ArgumentParser(description='Run Two-Phase Method model')
    parser.add_argument('--base_model', type=str, default='wmf', help='Base model name')
    parser.add_argument('--topk', type=int, default=100, help='Top K items to consider')
    parser.add_argument('--data_type', type=str, default='mock', help='Type of data to use')
    parser.add_argument('--target_year', type=str, default='period', help='Target year for the data')
    parser.add_argument('--number', type=int, default=0, help='discrete number')


    # Parse arguments
    args = parser.parse_args()
    
    # 1번 gpu 사용
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Use arguments
    base_model = args.base_model
    topk = args.topk
    data_type = args.data_type
    target_year = args.target_year
    number = args.number
    model_name = f"twophase_{base_model}"


    # base_model = "wmf"
    # topk = 100
    # model_name = f"twophase_{base_model}"
    # data_type = "mock"
    # target_year = 2023

    holdings_data, factor_params = get_data(data_type, target_year)
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]

    base_path = f"results/{target_year}/{base_model}_{number}"
    save_path = f"results/{target_year}/{model_name}_{number}"
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_model(base_path)
    assert m == model.n_users
    assert n == model.n_items
    total_score = []
    for user in range(m):
        total_score.append(
            model.forward(user, np.arange(model.n_items))
        )
    total_score = np.array(total_score)
    result = two_phase_method(total_score, holdings_data, factor_params, phase1_k=topk, is_test=False)
    save_pickle(result, os.path.join(save_path, "total_score.pkl"))
