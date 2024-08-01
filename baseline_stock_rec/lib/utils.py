import os
import pickle
import numpy as np
from lib.path import root_path
from joblib import Parallel, delayed

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

__all__ = [
    "check_stop_flag",
    "save_pickle",
    "load_pickle",
    "recompute_factors_batched",
    "get_data",
    "get_mean_variance",
    "get_correlation",
    "get_ret_data"
]


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, X_reg, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(range(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + 1).dot(Y_u)  # wegihted MF 의 weight 때문에

        if X_reg is not None:
            A += X_reg[k]

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR  # wegihted MF 의 weight 때문에

        A_stack[ib] = A
        B_stack[ib] = B
    # solve B_stack * X = A_stack
    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, lambda_reg, X=None,
                              dtype='float32', batch_size=2000, n_jobs=20):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)
    if X is not None:
        X_reg = lambda_reg * X
    else:
        X_reg = None
    X_new = np.zeros((m, f), dtype=dtype)

    num_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, X_reg, YTYpR,
                                                       batch_size, m, f, dtype)
                                  for b in range(num_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new


def check_stop_flag(dirpath):
    load_path = None
    stop_flag = False

    tmp = os.listdir(dirpath)
    tmp = [x for x in tmp if x.startswith("epoch_")]
    order = [int(x.split("epoch_")[1].split(".pkl")[0]) for x in tmp]
    order = np.argsort(order)
    tmp = list(np.array(tmp)[order])

    if not len(tmp) == 0:
        load_path = os.path.join(dirpath, tmp[-1])

        data = load_pickle(load_path)
        list_stop_cri = data["list_stop_cri"][-30:]
        if len(list_stop_cri) >= 30 and (
                list_stop_cri[0] <= min(list_stop_cri) or np.all(max(abs(np.diff(list_stop_cri))) < 1)):
            stop_flag = True

    return stop_flag, load_path


def save_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)
    f.close()


def load_pickle(path):
    # with open(path, "rb") as f:
    #     data = pickle.load(f)
    # f.close()
    data = np.load(path, allow_pickle=True)
    return data


def get_data(data_type, year):
    data_dir_name = os.path.join(root_path, "data")

    holdings_data = load_pickle(
        os.path.join(data_dir_name, '{}/{}/holdings_data.pkl'.format(data_type, year)))
    factor_params = load_pickle(
        os.path.join(data_dir_name, '{}/{}/factor_model_params.pkl'.format(data_type, year)))
    return holdings_data, factor_params


def get_ret_data(data_type, year):
    data_dir_name = os.path.join(root_path, "data")

    ret_data = load_pickle(
        os.path.join(data_dir_name, '{}/{}/weekly_ret_data.pkl'.format(data_type, year)))
    return ret_data


def get_mean_variance(factor_params):
    beta = factor_params["beta"]
    mu = np.matmul(factor_params["factor_mean"], beta.T)
    sig2_factor = factor_params["factor_variance"]
    sig2_eps = np.diag(factor_params["sig_eps_square"])
    cov = np.matmul(np.matmul(beta, sig2_factor), beta.T) + sig2_eps
    return mu, cov


def get_correlation(factor_params):
    _, cov = get_mean_variance(factor_params)

    diag = np.sqrt(np.diag(cov))
    invdiag = diag ** -1
    correlation = np.clip(invdiag.reshape(-1, 1) * cov * invdiag, -1, 1)
    return correlation
