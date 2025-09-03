import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer
import csv
from tppr_tgn.TPPR_Score import tppr_main

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True,
               train_data_tppr_scores=[],
               val_data_tppr_scores=[],
               test_data_tppr_scores=[],
               new_node_val_data_tppr_scores=[],
               new_node_test_data_tppr_scores=[]):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device
    self.train_data_tppr_scores = train_data_tppr_scores
    self.val_data_tppr_scores = val_data_tppr_scores
    self.test_data_tppr_scores = test_data_tppr_scores
    self.new_node_val_data_tppr_scores = new_node_val_data_tppr_scores
    self.new_node_test_data_tppr_scores = new_node_test_data_tppr_scores

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, batch_number=0, data_type='train_data'):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    if(data_type == 'train_data'):
      self.tppr_scores = self.train_data_tppr_scores
    elif (data_type == 'val_data'):
      self.tppr_scores = self.val_data_tppr_scores
    elif (data_type == 'test_data'):
      self.tppr_scores = self.test_data_tppr_scores
    elif (data_type == 'new_node_val_data'):
      self.tppr_scores = self.new_node_val_data_tppr_scores
    elif (data_type == 'new_node_test_data'):
      self.tppr_scores = self.new_node_test_data_tppr_scores

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:

      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors, batch_number=batch_number,
                                                           data_type=data_type)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      # neighbor embeddings
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors, batch_number=batch_number,
                                                   data_type=data_type)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      neighbor_nodes = neighbors_torch.cpu().numpy()
      tppr_dict = self.tppr_scores[batch_number]
      
      batch_size, n_neighbors, emb_dim = neighbor_embeddings.shape

      # Build TPPR weights tensor
      tppr_weights = torch.zeros(batch_size, n_neighbors, device=neighbor_embeddings.device)

      for i, src in enumerate(source_nodes):
          for j, nbr in enumerate(neighbor_nodes[i]):
              tppr_weights[i, j] = tppr_dict.get(int(src), {}).get(int(nbr), 0.0)

      # Weighted sum of neighbors
      weighted_neighbors = (tppr_weights.unsqueeze(-1) * neighbor_embeddings).sum(dim=1)  # [B, D]

      # Blend with source embedding
      source_embedding = 0.9 * source_node_conv_embeddings + 0.1 * weighted_neighbors

      source_embedding = self.aggregate(n_layers, source_embedding,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask,
                                        source_nodes,
                                        neighbors_torch.cpu().numpy(),
                                        weighted_neighbors
        )
      # The Final embedding for each epoch calculate here and send for train tgn model      
      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True,
               train_data_tppr_scores=[],
               val_data_tppr_scores=[],
               test_data_tppr_scores=[],
               new_node_val_data_tppr_scores=[],
               new_node_test_data_tppr_scores=[]):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory,
                                                  train_data_tppr_scores,
                                                  val_data_tppr_scores,
                                                  test_data_tppr_scores,
                                                  new_node_val_data_tppr_scores,
                                                  new_node_test_data_tppr_scores)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask, source_nodes, neighbor_nodes, weighted_neighbors):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    # Blend with source embedding
    source_embedding = 0.95 * source_embedding + 0.05 * weighted_neighbors

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True, num_batch=0,
                         train_data=[],
                         val_data=[],
                         test_data=[],
                         new_node_val_data=[],
                         new_node_test_data=[]):
  
  train_data_tppr_scores = tppr_scores_for_graph(data=train_data, batch_size=num_batch, data_type='train_data')
  val_data_tppr_scores = tppr_scores_for_graph(data=val_data, batch_size=num_batch, data_type='val_data')
  test_data_tppr_scores = tppr_scores_for_graph(data=test_data, batch_size=num_batch, data_type='test_data')
  new_node_val_data_tppr_scores = tppr_scores_for_graph(data=new_node_val_data, batch_size=num_batch, data_type='new_node_val_data')
  new_node_test_data_tppr_scores = tppr_scores_for_graph(data=new_node_test_data, batch_size=num_batch, data_type='new_node_test_data')
  
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                    train_data_tppr_scores=train_data_tppr_scores,
                                    val_data_tppr_scores=val_data_tppr_scores,
                                    test_data_tppr_scores=test_data_tppr_scores,
                                    new_node_val_data_tppr_scores=new_node_val_data_tppr_scores,
                                    new_node_test_data_tppr_scores=new_node_test_data_tppr_scores)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


def tppr_scores_for_graph(data, batch_size, data_type='train_data'):
  sources = data.sources
  destinations = data.destinations
  timestamps = data.timestamps
  edge_idxs = data.edge_idxs
  
  if(data_type == 'train_data'):
    labels = data.labels

  tppr_scores = []

  n_samples = len(sources)
  n_batches = (n_samples + batch_size - 1) // batch_size 

  for j in range(n_batches):
    start_idx = 0
    end_idx = min((j + 1) * batch_size, n_samples)

    filename = f"./tppr/wikipedia_{data_type}_batch_{j}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        if(data_type == 'train_data'):
          writer.writerow(["source", "destination", "timestamp", "edge_idx", "label"])
          for i in range(start_idx, end_idx):
            writer.writerow([sources[i], destinations[i], timestamps[i], edge_idxs[i], labels[i]])
        else:
          writer.writerow(["source", "destination", "timestamp", "edge_idx"])
          for i in range(start_idx, end_idx):
            writer.writerow([sources[i], destinations[i], timestamps[i], edge_idxs[i]])
        
    tppr_scores.append(tppr_main(filename))

  return tppr_scores