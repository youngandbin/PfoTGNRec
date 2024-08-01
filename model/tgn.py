import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from model.feature_encoding import UserEncode, ItemEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, user_features, item_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, 
               message_dimension=100, memory_dimension=500,
               embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    ####################################MLP넣어주기#######################
    self.user_encoder = UserEncode(user_features.shape[1])
    self.item_encoder = ItemEncode(item_features.shape[1])

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)

    # nofrmalize edge_features
    edge_features = edge_features.astype(np.float32)
    edge_features -= edge_features.mean(axis=0)
    edge_features /= edge_features.std(axis=0)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
    
    self.n_node_features = self.node_raw_features.shape[1] 
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features   = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features) # dimension=self.n_node_features
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2*self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 user_features=user_features,
                                                 item_features=item_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 user_encoder = self.user_encoder, ##################MLP
                                                 item_encoder = self.item_encoder, ##################MLP
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)


  def compute_temporal_embeddings_p(self, source_nodes, destination_nodes, 
                                    p_pos_nodes, p_neg_nodes, 
                                    edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    :param source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
    bs = len(source_nodes)
    NUM_POS_TRAIN = int(p_pos_nodes.shape[0]/bs) # 1
    NUM_NEG_TRAIN = int(p_neg_nodes.shape[0]/bs) # 3

    nodes = np.concatenate([source_nodes, destination_nodes, p_pos_nodes, p_neg_nodes]) ## yj
    positives = np.concatenate([source_nodes, destination_nodes])
    edge_times_duplicated = [x for x in edge_times for _ in range(NUM_NEG_TRAIN)]
    timestamps = np.concatenate([edge_times, edge_times, edge_times, edge_times_duplicated])

    """
    1. Memory update (RNN module)
    """

    # Even when moving to the next batch, the TGN memory retains the memory from the previous batch
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches ( s = RNN(m, s-) )
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages) # Dictionary where the keys are node IDs and the values are messages

      else: # not executed
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      p_pos_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[p_pos_nodes].long()
      p_pos_time_diffs = (p_pos_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      p_neg_time_diffs = torch.LongTensor(edge_times_duplicated).to(self.device) - last_update[p_neg_nodes].long()
      p_neg_time_diffs = (p_neg_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, p_pos_time_diffs, p_neg_time_diffs], dim=0)


    """
    2. Embedding (GNN module)
    """
    
    # Compute the embeddings using the embedding module 
    # shape: torch.Size([3*bs, dim]) # bs (batch size) for each source, destination, and negative samples respectively.
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs) 

    # source node 1개당 p_pos_node 1개, p_neg_node 3개 있음. 따라서 2*bs, 3*bs로 나눠줌.
    n = 1
    source_node_embedding       = node_embedding[:n*bs]
    destination_node_embedding  = node_embedding[n*bs : (n+1)*bs]
    p_pos_node_embedding        = node_embedding[(n+1)*bs : (n+1+NUM_POS_TRAIN)* bs] ## yj
    p_neg_node_embedding        = node_embedding[(n+1+NUM_POS_TRAIN)*bs:] ## yj

    """
    3. Raw message store
    """
    
    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations 
        # (since now we have new messages for them)
        # assign values to self.memory
        self.update_memory(positives, self.memory.messages) # s = RNN(m, s-)
        
        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)

      if self.memory_update_at_start: 
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else: # not executed
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        p_pos_node_embedding = memory[p_pos_nodes]
        p_neg_node_embedding = memory[p_neg_nodes]

    return source_node_embedding, destination_node_embedding, p_pos_node_embedding, p_neg_node_embedding
    
  def compute_temporal_embeddings(self, source_nodes, destination_nodes, p_neg_nodes, 
                                        edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    :param source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, p_neg_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    size = int(len(p_neg_nodes)/len(source_nodes)) # because negatives_batch.flatten() = (BATCH_SIZE * size,)
    edge_times_duplicated = [x for x in edge_times for _ in range(size)] 
    timestamps = np.concatenate([edge_times, edge_times, edge_times_duplicated])

    """
    1. Memory update (RNN module)
    """

    # Even when moving to the next batch, the TGN memory retains the memory from the previous batch
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches ( s = RNN(m, s-) )
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages) # Dictionary where the keys are node IDs and the values are messages

      else: # not executed
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      p_neg_time_diffs = torch.LongTensor(edge_times_duplicated).to(self.device) - last_update[p_neg_nodes].long() 
      p_neg_time_diffs = (p_neg_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, p_neg_time_diffs], dim=0) #####error 

    """
    2. Embedding (GNN module)
    """
    
    # Compute the embeddings using the embedding module 
    # shape: torch.Size([3*bs, dim]) # bs (batch size) for each source, destination, and negative samples respectively.
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs) 

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples : 2 * n_samples]
    p_neg_node_embedding = node_embedding[2 * n_samples :]

    """
    3. Raw message store
    """
    
    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations 
        # (since now we have new messages for them)
        # assign values to self.memory
        self.update_memory(positives, self.memory.messages) # s = RNN(m, s-)

        # Check if two tensors are identical.
        # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
        #   "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)

      if self.memory_update_at_start: 
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else: # not executed
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        p_neg_node_embedding = memory[p_neg_nodes]

    return source_node_embedding, destination_node_embedding, p_neg_node_embedding

  def update_memory(self, nodes, messages):

    # Aggregate messages for the same nodes ('last')
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, 
                                      unique_messages, 
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):

    # Aggregate messages for the same nodes ('last')
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # memory_updater.get_updated_memory: Update the existing memory using the current memory.
    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)
    return updated_memory, updated_last_update


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features, source_time_delta_encoding], dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder