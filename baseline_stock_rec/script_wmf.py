import os
import argparse
from lib.path import root_path
from lib.utils import get_data
from lib.models.base_svd import SVD_als
import pickle

# data_type = "mock"
# target_year = 2023
# gamma = 3
# latent_dim = 30
# reg_param = 0.001

# Set up argparse
parser = argparse.ArgumentParser(description='Run SVD ALS model')
parser.add_argument('--data_type', type=str, default='mock', help='Type of data to use')
parser.add_argument('--target_year', type=str, default='period', help='Target year for the data')
parser.add_argument('--latent_dim', type=int, default=30, help='Dimension of latent factors')
parser.add_argument('--reg_param', type=float, default=0.001, help='Regularization parameter')
parser.add_argument('--alpha', type=float, default=10, help='confidence level')
parser.add_argument('--number', type=int, default=0, help='discrete number')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')

# Parse arguments
args = parser.parse_args()

# 1번 gpu 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use arguments
data_type = args.data_type
target_year = args.target_year
latent_dim = args.latent_dim
reg_param = args.reg_param
number = args.number
alpha = args.alpha
epochs = args.epochs

holdings_data, factor_params = get_data(data_type, target_year)


dirpath = os.path.join(root_path, "results/{}/wmf_{}".format(target_year, number))
print(dirpath)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# save all the args to dictionary
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
# save the args to pickle
dirpath2 = os.path.join(root_path, "results/{}/wmf_{}".format(target_year,number))
if not os.path.exists(dirpath2):
    os.makedirs(dirpath2)
with open(os.path.join(dirpath2, "parameters.pkl"), "wb") as f:
    pickle.dump(args_dict, f)


mf = SVD_als(
    holdings_data, n_epochs=epochs, n_factors=latent_dim, alpha=alpha,
    reg_param=reg_param, tmp_save_path=dirpath,
)
mf.fit()
