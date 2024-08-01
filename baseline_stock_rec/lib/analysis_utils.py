import os
import re

from lib.models.base_svd import SVD
from lib.utils import check_stop_flag


def get_model(result_path):
    converged, load_path = check_stop_flag(result_path)
    mf = SVD(None)
    mf.tmp_load_path = load_path
    mf.load_variables()
    return mf


def get_name(model, data_type, target_year, model_type, index_name):
    hyper_param_list = []
    for hyper_param in index_name:
        if hyper_param == "data_type":
            hyper_param_list.append(data_type)
        elif hyper_param == "year":
            hyper_param_list.append(target_year)
        elif hyper_param == "model":
            hyper_param_list.append(model_type)
        else:
            if model is not None and hasattr(model, hyper_param):
                val = getattr(model, hyper_param)
                if val is None:
                    hyper_param_list.append(0.)
                else:
                    hyper_param_list.append(float(val))
            else:
                hyper_param_list.append(0.)
    return tuple(hyper_param_list)
