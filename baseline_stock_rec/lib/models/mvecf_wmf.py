import numpy as np
from lib.models.base_svd import SVD
from sklearn.metrics import average_precision_score
import os
from joblib import Parallel, delayed
from tqdm import tqdm

os.environ['OPENBLAS_NUM_THREADS'] = '4'
__all__ = [
    "MVECF_WMF",
]


class MVECF_WMF(SVD):
    def __init__(self, data, factor_params, n_factors=30, n_epochs=100, lr=0.001, reg_param=0.001, reg_param2=0.001,
                 early_stop=True, alpha=10, batch_size=128, reg_param_mv=None, gamma=None,
                 tmp_save_path=None, tmp_load_path=None, ):
        super(MVECF_WMF, self).__init__(data=data, n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg_param=reg_param,
                                        early_stop=early_stop, alpha=alpha, batch_size=batch_size,
                                        tmp_save_path=tmp_save_path, tmp_load_path=tmp_load_path, )
        self.factor_params = factor_params

        self.reg_param_mv = reg_param_mv
        if reg_param2 is None:
            reg_param2 = self.reg_param
        self.reg_param2 = reg_param2
        self.gamma = gamma
        self.val_loss_mv = 0
        self.train_loss_mv = 0

        sig2_M = self.factor_params["factor_variance"]
        beta = self.factor_params["beta"]
        sig2_eps = self.factor_params["sig_eps_square"]
        Sigma = np.matmul(np.matmul(beta, sig2_M), beta.T) + np.diag(sig2_eps)
        self.diag_Sigma = np.diag(Sigma)
        self.off_diag_Sigma = Sigma - np.diag(self.diag_Sigma)

        self.mu_items = self.cal_item_means()

        # generate dense rating and weight
        self.dense_rating = []
        self.dense_weight = []
        self.avg_true_beta = []
        self.num_holding = []
        for user in tqdm(range(self.n_users)):
            index_start = self.train_indptr[user]
            index_end = self.train_indptr[user + 1]
            train_items = self.train_data[1][index_start:index_end]
            train_ratings = self.train_data[2][index_start:index_end]
            self.num_holding.append(index_end - index_start)
            if self.num_holding[user] == 0:
                self.dense_rating.append(np.zeros(self.n_items))
                self.dense_weight.append(np.zeros(self.n_items))
                self.avg_true_beta.append(np.zeros(5))
                continue
            self.avg_true_beta.append(beta[train_items].mean(axis=0))
   
            
            i_user, r_user, w_user = self._get_total_items_ratings_init(user, train_items, train_ratings)
            self.dense_rating.append(r_user)
            self.dense_weight.append(w_user)


        self.num_holding = np.array(self.num_holding)
        print("user_num ", self.n_users, " item_num ", self.n_items, " factor_num ", self.n_factors)
        print("csr_train", self.csr_train.toarray().shape, " off_diag_Sigma", self.off_diag_Sigma.shape)
        self.sum_true_cov = np.matmul(self.csr_train.toarray(), self.off_diag_Sigma)
        self.avg_true_beta = np.array(self.avg_true_beta)

        self.dense_rating = np.array(self.dense_rating)
        self.dense_rating = self.dense_rating - self.dense_rating.mean()
        self.dense_weight = np.array(self.dense_weight)
        self.params_not_save += [
            "factor_params", "diag_Sigma", "off_diag_Sigma", "mu_items", "dense_rating", "dense_weight",
            "num_holding", "sum_true_cov", "avg_true_beta"
        ]

    def cal_item_means(self):
        return np.matmul(self.factor_params["beta"], self.factor_params["factor_mean"])

    def _get_total_items_ratings_init(self, user, train_items, train_ratings):
        beta = self.factor_params["beta"]
        sig2_M = self.factor_params["factor_variance"]
        iter_train_items_raitings = iter(zip(train_items, train_ratings))

        tmp_item = []
        tmp_rating = []
        tmp_weight = []
        # initialize
        i_train, r_train = next(iter_train_items_raitings)
        # start
        for i, mu_i, beta_i, sig2_i in zip(range(self.n_items), self.mu_items, beta, self.diag_Sigma):
            prev_weight = 1
            prev_rating = 0
            if i == i_train:
                prev_weight = self.alpha
                prev_rating = r_train
                try:
                    i_train, r_train = next(iter_train_items_raitings)
                except StopIteration:
                    i_train = self.n_items
                    r_train = 0
            mv_weight = self.gamma / 2 * self.reg_param_mv * sig2_i
            
            avg_true_beta = self.avg_true_beta[user] - beta_i / self.num_holding[user] * prev_rating
            mv_rating = (mu_i / self.gamma - (beta_i * np.matmul(avg_true_beta, sig2_M) / 2).sum()) / sig2_i
            weight = prev_weight + mv_weight
            rating = (prev_weight * prev_rating + mv_weight * mv_rating) / weight

            tmp_item.append(i)
            tmp_rating.append(rating)
            tmp_weight.append(weight)

        return np.array(tmp_item).astype(int), np.array(tmp_rating).astype(np.float32), np.array(tmp_weight).astype(
            np.float32)

    def _get_total_items_ratings(self, user, train_items, train_ratings):
        return np.arange(self.n_items), self.dense_rating[user], self.dense_weight[user]

    def calculate_valloss(self):
        print("calculating validation loss")
        # validation
        beta = self.factor_params["beta"]
        sig2_M = self.factor_params["factor_variance"]

        self.val_loss = 0
        self.map_valid = 0
        
        pass_num = 0
        # weight and rating for validation set are not modified in self.__init__
        for user in range(self.n_users):
            if self.valid_indptr[user] == self.valid_indptr[user+1]:
                pass_num += 1
                continue
            if self.num_holding[user] == 0:
                pass_num += 1
                continue
            index = range(self.valid_indptr[user], self.valid_indptr[user + 1])
            self.valid_data = np.array(self.valid_data)
            items = self.valid_data[1][index]
            # items = self.valid_data[1][self.valid_indptr[user]:self.valid_indptr[user + 1]]

            beta_i = beta[items]
            mu_i = self.mu_items[items].reshape(-1, 1)
            sig2_i = self.diag_Sigma[items].reshape(-1, 1)

            prev_rating = self.valid_data[2][index].reshape(-1, 1)
            prev_weight = (1 - prev_rating) + prev_rating * self.alpha
            
            avg_true_beta = self.avg_true_beta[user] - beta_i / self.num_holding[user] * prev_rating
            mv_weight = self.gamma / 2 * self.reg_param_mv * sig2_i
            mv_rating = (mu_i / self.gamma
                         - (beta_i * np.matmul(avg_true_beta, sig2_M) / 2).sum(axis=1, keepdims=True)) / sig2_i
            weight = prev_weight + mv_weight
            rating = (prev_weight * prev_rating + mv_weight * mv_rating) / weight

            weight = weight.flatten()
            rating = rating.flatten()
            estimate = np.matmul(self.pu[user], self.qi[items].T)
            self.val_loss += (weight * (rating - estimate) ** 2).sum()

            y_true = prev_rating
            y_score = estimate

            self.map_valid += average_precision_score(y_true, y_score)

        self.map_valid = self.map_valid / self.n_users
        print('total_user', self.n_users)
        print('pass_num', pass_num)
        print("validation MAP: {}".format(self.map_valid))

    def fit_epoch(self, shuffle=True):
        lambda_pu_reg = self.reg_param
        lambda_qi_reg = self.reg_param2
        dtype = 'float32'

        S = self.dense_weight - 1

        self.pu = recompute_factors_batched(self.qi, S, self.dense_rating, lambda_pu_reg, dtype=dtype)
        self.qi = recompute_factors_batched(self.pu, S.T, self.dense_rating.T, lambda_qi_reg, dtype=dtype)

        print("calculating train loss")
        r_predict = np.matmul(self.pu, self.qi.T)
        reg_loss = self.reg_param * ((self.pu ** 2).sum() + (self.qi ** 2).sum())
        self.train_loss = (self.dense_weight * (self.dense_rating - r_predict) ** 2).sum() + reg_loss

        dense_train_data = self.csr_train.toarray()
        rec_weight = (1 - dense_train_data) + dense_train_data * self.alpha
        self.train_loss_rec = (rec_weight * (dense_train_data - r_predict) ** 2).sum()
        self.train_loss_mv = self.cal_mv_loss(r_predict)

    def cal_mv_loss(self, estimate_ui):
        mu_i = np.matmul(self.factor_params["beta"], self.factor_params["factor_mean"])
        loss_mv = self.gamma / 2 * (
                estimate_ui ** 2 * self.diag_Sigma
                + estimate_ui * self.sum_true_cov / self.num_holding.reshape(-1, 1)
        ) - mu_i * estimate_ui
        return loss_mv.sum()


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, X_reg, YTYpR, rating, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(range(lo, hi)):
        s_u = S[k]
        p_u = rating[k]
        A = ((s_u + 1) * p_u).dot(Y)  # YTCup(u)

        if X_reg is not None:
            A += X_reg[k]

        YTSY = np.dot(Y.T, (Y * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B
    # solve B_stack * X = A_stack
    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, rating, lambda_reg, X=None,
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

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, X_reg, YTYpR, rating,
                                                       batch_size, m, f, dtype)
                                  for b in range(num_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new
