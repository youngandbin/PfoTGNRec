import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
import time
from copy import deepcopy
from tqdm import tqdm
from lib.utils import save_pickle, load_pickle, recompute_factors_batched
import os
import gc
import inspect
from scipy.sparse import csr_matrix

__all__ = [
    "SVD",
    "SVD_als"
]


class SVD(object):
    def __init__(self, data, n_factors=30, n_epochs=100, lr=0.001, reg_param=0.001, verbose=1,
                 early_stop=True, alpha=10, batch_size=128, tmp_save_path=None, tmp_load_path=None, ):
        self.verbose = verbose
        if data is not None:
            self.train_data = data["train_data"]
            self.train_indptr = data["train_indptr"]
            self.valid_data = data["valid_data"]
            self.valid_indptr = data["valid_indptr"]
            self.n_users = data["n_users"]
            self.n_items = data["n_items"]
            self.csr_train = csr_matrix(
                (self.train_data[2], (self.train_data[0], self.train_data[1])), shape=(self.n_users, self.n_items)
            )  # to calculate train error fast
        else:
            self.train_data = None
            self.train_indptr = None
            self.n_users = 0
            self.n_items = 0
            self.csr_train = None
        self.list_stop_cri = []
        self.early_stop = early_stop

        self.tmp_save_path = tmp_save_path
        self.tmp_load_path = tmp_load_path

        self.start_epoch = 0
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_param = reg_param

        self.batch_size = batch_size
        self.alpha = alpha  # confidence weight

        self.pu = None
        self.qi = None

        self.train_loss = 0
        self.train_loss_rec = 0
        self.val_loss = 0
        self.val_loss_rec = 0
        self.map_valid = 0

        self.params_not_save = [
            "start_epoch", "train_data", "test_data", "valid_data", "valid_indptr", "train_indptr", "test_indptr",
            "tmp_load_path", "tmp_save_path", "n_epochs", "early_stop", "csr_train"
        ]

    def __call__(self, rows, cols):
        return self.forward(rows, cols)

    def forward(self, rows, cols):
        return np.sum(self.pu[rows] * self.qi[cols], axis=1)

    def init_params(self):
        self.list_stop_cri = []
        if self.pu is None:
            self.pu = np.random.normal(0, 0.01, size=(self.n_users, self.n_factors))
        if self.qi is None:
            self.qi = np.random.normal(0, 0.01, size=(self.n_items, self.n_factors))

    def save_variables(self, finished_epoch):
        if self.tmp_save_path is not None:
            if not os.path.exists(self.tmp_save_path):
                os.makedirs(self.tmp_save_path)
            next_start_epoch = finished_epoch + 1
            attr_list = [("start_epoch", next_start_epoch)]
            for i in inspect.getmembers(self):
                if not i[0].startswith('_'):
                    if not inspect.ismethod(i[1]):
                        if not i[0] in self.params_not_save:
                            attr_list.append(i)
            variable_dict = {}
            for key, item in attr_list:
                variable_dict[key] = item
            save_pickle(variable_dict, os.path.join(self.tmp_save_path, "epoch_{}.pkl".format(finished_epoch)))

    def update(self, user, item, rating, weight):
        puf = deepcopy(self.pu[user])
        qif = deepcopy(self.qi[item])
        err = rating - np.sum(puf * qif, axis=1)
        # update factors
        self.pu[user] += self.lr * ((weight * err).reshape((-1, 1)) * qif - self.reg_param * puf)
        self.qi[item] += self.lr * ((weight * err).reshape((-1, 1)) * puf - self.reg_param * qif)

        self.train_loss_rec += (weight * err ** 2).sum()

    def load_variables(self):
        if self.tmp_load_path is not None:
            variable_dict = load_pickle(self.tmp_load_path)
            for key, value in variable_dict.items():
                if key != "n_epochs" or key != "early_stop":
                    setattr(self, key, value)

    def _get_total_items_ratings(self, user, train_items, train_ratings):
        iter_train_items_raitings = iter(zip(train_items, train_ratings))

        tmp_item = []
        tmp_rating = []
        tmp_weight = []
        # initialize
        i_train, r_train = next(iter_train_items_raitings)
        # start
        for i in range(self.n_items):
            item = i
            rating = 0
            weight = 1
            if i == i_train:
                item = i
                rating = r_train
                weight = self.alpha
                try:
                    i_train, r_train = next(iter_train_items_raitings)
                except StopIteration:
                    i_train = self.n_items
                    r_train = 0
            tmp_item.append(item)
            tmp_rating.append(rating)
            tmp_weight.append(weight)

        return np.array(tmp_item).astype(int), np.array(tmp_rating).astype(np.float32), np.array(tmp_weight).astype(
            np.float32)

    def gen_rating_pairs(self, user_range, shuffle=True):
        i_total = []
        r_total = []
        u_total = []
        w_total = []

        time.sleep(0.1)
        for u in user_range:
            index_start = self.train_indptr[u]
            index_end = self.train_indptr[u + 1]
            train_items = self.train_data[1][index_start:index_end]
            train_ratings = self.train_data[2][index_start:index_end]

            i_user, r_user, w_user = self._get_total_items_ratings(u, train_items, train_ratings)
            u_total.append(np.ones(len(i_user)) * u)
            i_total.append(i_user)
            r_total.append(r_user)
            w_total.append(w_user)

        u_total = np.concatenate(u_total).astype(int)
        i_total = np.concatenate(i_total).astype(int)
        r_total = np.concatenate(r_total).astype(np.float32)
        w_total = np.concatenate(w_total).astype(np.float32)

        gc.collect()
        if shuffle:
            index = np.arange(len(i_total))
            np.random.shuffle(index)

            num_batch = int(len(index) / self.batch_size)
            u_total = np.array_split(u_total[index], num_batch)
            i_total = np.array_split(i_total[index], num_batch)
            r_total = np.array_split(r_total[index], num_batch)
            w_total = np.array_split(w_total[index], num_batch)

        return u_total, i_total, r_total, w_total

    def fit_epoch(self, shuffle=True):
        self.train_loss = 0
        self.train_loss_rec = 0

        num_partition = 3
        total_user_range = np.arange(self.n_users)
        np.random.shuffle(total_user_range)
        for partition in range(num_partition):
            if self.verbose:
                print("generating rating partition {}".format(partition))
            start_idx = partition * int(self.n_users / num_partition)
            if partition == num_partition - 1:
                end_idx = self.n_users
            else:
                end_idx = (partition + 1) * int(self.n_users / num_partition)
            user_range = total_user_range[start_idx:end_idx]
            u_total, i_total, r_total, w_total = self.gen_rating_pairs(user_range, shuffle=shuffle)
            if self.verbose:
                print("updating rating parameters..")
            time.sleep(0.1)
            for batch in tqdm(range(len(i_total))):
                batch_u = u_total[batch]
                batch_i = i_total[batch]
                batch_r = r_total[batch]
                batch_w = w_total[batch]
                self.update(batch_u, batch_i, batch_r, batch_w)

        r_predict = np.matmul(self.pu, self.qi.T)
        reg_loss = self.reg_param * ((self.pu ** 2).sum() + (self.qi ** 2).sum())
        dense_train_data = self.csr_train.toarray()
        rec_weight = (1 - dense_train_data) + dense_train_data * self.alpha
        self.train_loss_rec = (rec_weight * (dense_train_data - r_predict) ** 2).sum()
        self.train_loss = self.train_loss_rec + reg_loss
        return r_predict

    def check_early_stop(self):
        list_stop_cri = self.list_stop_cri[-30:]
        if len(list_stop_cri) >= 30 and (
                list_stop_cri[0] <= min(list_stop_cri) or abs(list_stop_cri[0]-list_stop_cri[-1]) < 1
        ):
            return list_stop_cri[0]
        else:
            return None

    def calculate_valloss(self):
        if self.verbose:
            print("calculating validation loss")
        self.val_loss = 0
        
        self.valid_data = np.array(self.valid_data)

        u_total = self.valid_data[0]
        i_total = self.valid_data[1]
        r_total = self.valid_data[2]
        w_total = (1 - self.valid_data[2]) + self.valid_data[2] * self.alpha
        pud = self.pu[u_total]
        qid = self.qi[i_total]
        r_predict = np.sum(pud * qid, axis=1)
        self.val_loss_rec = (w_total * (r_total - r_predict) ** 2).sum()
        self.val_loss += self.val_loss_rec
        self.map_valid = 0
        y_true = self.valid_data[2]
        y_score = r_predict
        for user in range(self.n_users):
            tmp_true = y_true[self.valid_indptr[user]:self.valid_indptr[user + 1]]
            if len(tmp_true) == 0:
                continue
            tmp_score = y_score[self.valid_indptr[user]:self.valid_indptr[user + 1]]
            self.map_valid += average_precision_score(tmp_true, tmp_score)
        self.map_valid = self.map_valid / self.n_users
        if self.verbose:
            print("validation MAP: {}".format(self.map_valid))
        return pud, qid, r_predict

    def fit(self):
        self.init_params()
        self.load_variables()
        for current_epoch in range(self.start_epoch, self.n_epochs + 1):
            start_time = time.time()
            if self.verbose:
                print("\nProcessing epoch {}".format(current_epoch))
                print(self.tmp_save_path)
            self.fit_epoch()
            if not np.isfinite(self.train_loss) or np.isnan(self.train_loss) or np.isnan(self.val_loss):
                break
            self.calculate_valloss()
            self.list_stop_cri.append(self.val_loss)
            elapsed_time = time.time() - start_time
            if self.verbose:
                print("\nelapsed time: {}, train loss: {}, validation loss: {}".format(
                    elapsed_time, self.train_loss, self.val_loss
                ))
            self.save_variables(current_epoch)
            if self.early_stop:
                flag = self.check_early_stop()
                if flag:
                    return flag
        return self.list_stop_cri[-1]


class SVD_als(SVD):
    def __init__(self, data, n_factors=30, n_epochs=100, lr=0.001, reg_param=0.001, verbose=1,
                 early_stop=True, alpha=10, batch_size=128, tmp_save_path=None, tmp_load_path=None,
                 reg_param2=None, ):
        super(SVD_als, self).__init__(data=data, n_factors=n_factors, n_epochs=n_epochs, verbose=verbose,
                                      lr=lr, early_stop=early_stop, reg_param=reg_param,
                                      alpha=alpha, batch_size=batch_size, tmp_save_path=tmp_save_path,
                                      tmp_load_path=tmp_load_path, )
        if reg_param2 is None:
            reg_param2 = reg_param
        self.reg_param2 = reg_param2

    def fit_epoch(self, shuffle=True):
        lambda_pu_reg = self.reg_param
        lambda_qi_reg = self.reg_param2
        dtype = 'float32'

        S = self.csr_train.copy()
        S.data = (self.alpha - 1) * S.data

        self.pu = recompute_factors_batched(self.qi, S, lambda_pu_reg, dtype=dtype)
        ST = S.T.tocsr()
        self.qi = recompute_factors_batched(self.pu, ST, lambda_qi_reg, dtype=dtype)

        r_predict = np.matmul(self.pu, self.qi.T)
        reg_loss = self.reg_param * ((self.pu ** 2).sum() + (self.qi ** 2).sum())
        dense_train_data = self.csr_train.toarray()
        rec_weight = (1 - dense_train_data) + dense_train_data * self.alpha
        self.train_loss_rec = (rec_weight * (dense_train_data - r_predict) ** 2).sum()
        self.train_loss = self.train_loss_rec + reg_loss
