import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle


def preprocess(data_name):
  """
  transaction.csv -> df
  """
  u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []
  feat_l = []
  portfolio_l = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):

      e = line.strip().split(',')

      u = int(e[0])
      i = int(e[1])
      ts = float(e[2])
      label = e[3] # float(e[3])  
      feat = float(e[4]) 
      portfolio = e[5:] # e.g., 281740,011150,108860 (stock code)
      portfolio = [x.replace("'", "").replace('"', '') for x in portfolio] # remove ' and " from list
      # portfolio = [int(x) for x in portfolio] # converting a string to an int results in a loss of the original 6-digit format

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      portfolio_l.append(portfolio)
      feat_l.append(feat)

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list,
                       'portfolio': portfolio_l}), np.array(feat_l)


def reindex(df, bipartite=True):

  new_df = df.copy()

  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    # Original: Both user and item start from 0.

    # item idx starts from the end of user idx
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    # Add 1 to both user and item indices.
    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, period='1'):
  """
  transaction.csv -> transaction.json
  """

  save_path = './data/period_{}/'.format(period)

  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = save_path+'{}.csv'.format(data_name)
  OUT_DF = save_path+'ml_{}.json'.format(data_name)
  OUT_FEAT = save_path+'ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = save_path+'ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  # # edge features
  # empty = np.zeros(len(feat))[np.newaxis, :]
  # feat = np.vstack([empty, feat])

  # edge features
  feat = np.load(save_path+'{}.npy'.format(data_name))
  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])
  
  # node features
  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172)) 

  # save
  new_df.to_json(OUT_DF, orient='records') # to preserve lists (portfolios) in dataframe
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
 

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='transaction')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--period', type=str, help='Period of the data (eg. year or month)', default='1')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite, period=args.period)