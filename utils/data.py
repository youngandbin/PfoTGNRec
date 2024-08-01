import numpy as np
import random
import pandas as pd


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, portfolios):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
    self.portfolios = portfolios

def get_data(dataset_name, period, data_split):

  save_path = f'./data/period_{period}/'

  ### Load data and train val test split
  graph_df = pd.read_json(save_path+'ml_{}.json'.format(dataset_name)) 
  edge_features = np.load(save_path+'ml_{}.npy'.format(dataset_name))
  node_features = np.load(save_path+'ml_{}_node.npy'.format(dataset_name)) 
  user_features = np.load(save_path+'ml_{}_user.npy'.format(dataset_name)) 
  item_features = np.load(save_path+'ml_{}_item.npy'.format(dataset_name)) 

  # convert data_split
  # e.g., [8,1,1] -> [0.8, 0.9], [6,2,2] -> [0.6, 0.8]
  train, valid, test = data_split
  assert train + valid + test == 10, "data split should sum to 10"
  print('data split: train, valid, test:', train, valid, test)
  train = train / 10.0
  valid = train + (valid / 10.0)

  init_time = graph_df.ts.min()
  val_time, test_time = list(np.quantile(graph_df.ts, [train, valid]))
  print('init time: ', init_time)
  print('val time: ', val_time)
  print('test time: ', test_time)

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  portfolios = graph_df.portfolio.values # an array of lists
  
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels, portfolios)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  """
  Not dividing the 'new nodes'
  """

  train_mask = timestamps <= val_time
  val_mask   = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask  = timestamps > test_time

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask], portfolios[train_mask])
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask], portfolios[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask], portfolios[test_mask])

  upper_u = graph_df.u.max()

  print('')
  print("full data: {} interactions, {} different nodes".format(
    full_data.n_interactions, full_data.n_unique_nodes))
  print("train data: {} interactions, {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("valid data: {} interactions, {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("test data: {} interactions, {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print('')

  return node_features, user_features, item_features, edge_features, full_data, train_data, val_data, test_data, upper_u


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

