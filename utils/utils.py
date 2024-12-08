import numpy as np
import torch

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, portfolio_list, upper_u, map_item_id, seed=None):
    """
    dst_list: item idx starting from upper_u+1
    portfolio_list: stock codes
    """
    self.seed = None
    self.src_list = src_list
    self.dst_unique = np.unique(dst_list)
    # 240518 
    self.portfolio_list = [                                 # stock code (6 digits) -> Index starting from 0
        [map_item_id[item] for item in sublist if item] 
        for sublist in portfolio_list
    ]
    # self.portfolio_list = [[map_item_id[item] for item in sublist] for sublist in portfolio_list]       # stock code (6 digits) -> Index starting from 0
    self.portfolio_list = [[item + upper_u + 1 for item in sublist] for sublist in self.portfolio_list] # Index starting from 0 -> item idx starting from upper_u+1

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    """
    Set seeds for validation and testing so negatives are the same across different runs
    NB: in the inductive setting, negatives are sampled only amongst other new nodes
    """
    
    dst_index = []
    for i, src in enumerate(self.src_list): # interaction loop 
      
      # 포트폴리오에 이미 가지고 있는 주식은 추천하지 않게 하기 위한 세팅
      available_dst = np.setdiff1d(self.dst_unique, self.portfolio_list[i])

      # If the number of batch item sets is less than the 'size'
      if len(available_dst) < size: 
        if self.seed is not None: # 평가
            sample_dst = self.random_state.choice(available_dst, size=size, replace=True)
            # 240601 샘플링하지 말고 그냥 사용 (-> interaction마다 가지고 있는 포트폴리오가 다르므로, 샘플링된 아이템들의 길이가 안 맞는 에러)
            # sample_dst = available_dst
        else: # 훈련
            sample_dst = np.random.choice(available_dst, size=size, replace=True)
      # If the number of batch item sets is greater than the 'size'
      else:
        if self.seed is not None:
            sample_dst = self.random_state.choice(available_dst, size=size, replace=False) 
        else:
            sample_dst = np.random.choice(available_dst, size=size, replace=False) 

      dst_index.append(sample_dst)
    return np.array(dst_index) 
        

def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, 
                                                      data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp)) # For example, if Node 1 has 3 neighbors, adj_list[1] = [(2, 0, 0), (3, 1, 1), (4, 2, 2)] 이런 식으로 저장됨
    adj_list[destination].append((source, edge_idx, timestamp)) # For example, if Node 10 has 2 neighbors, adj_list[10] = [(9, 3, 3), (11, 4, 4)] 이런 식으로 저장됨

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors])) # [neighbor1, neighbor2, ...] # In the above example, it would be [2, 3, 4].
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors])) # [edge_idx1, edge_idx2, ...] # In the above example, it would be [0, 1, 2].
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors])) # [timestamp1, timestamp2, ...] # In the above example, it would be [0, 1, 2].

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    # Find the position (index) of the reference ts.
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    # The first return value: Take only [:i] among the neighbor indices of the src_idx node."
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    # Iterate through each node..
    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):

      # Find neighbors here..
      # Determine neighbors based on the adj_list that stores the source and destination of the train data. Therefore, in the case of negative, neighbors are determined based on the destination node.
      # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)  

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors) 

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
          
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times