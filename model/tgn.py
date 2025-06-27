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

class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device,
                 tppr_embeddings=None,
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
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

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        # ---- TPPR embedding support ----
        if tppr_embeddings is not None:
            self.tppr_embeddings = torch.from_numpy(tppr_embeddings.astype(np.float32)).to(device)
            self.tppr_dim = self.tppr_embeddings.shape[1]
            self.use_tppr = True
            # Project [node_emb | tppr_emb] -> node_emb
            self.combination_layer = torch.nn.Linear(self.n_node_features + self.tppr_dim,
                                                     self.n_node_features)
        else:
            self.tppr_embeddings = None
            self.tppr_dim = 0
            self.use_tppr = False

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
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
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    def get_combined_embedding(self, node_ids, tgn_embeddings):
        if self.use_tppr:
            tppr_embeds = self.tppr_embeddings[node_ids]
            combined = torch.cat([tgn_embeddings, tppr_embeds], dim=1)
            return self.combination_layer(combined)
        else:
            return tgn_embeddings

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                   dim=0)

        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        node_indices = torch.from_numpy(nodes).to(self.device)
        node_embedding = self.get_combined_embedding(node_indices, node_embedding)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                self.update_memory(positives, self.memory.messages)
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
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        source_embedding, destination_embedding, negative_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
        pos_prob = torch.sigmoid(self.affinity_score(source_embedding, destination_embedding))
        neg_prob = torch.sigmoid(self.affinity_score(source_embedding, negative_embedding))
        return pos_prob, neg_prob

    def set_neighbor_finder(self, neighbor_finder):
        self.embedding_module.neighbor_finder = neighbor_finder
        self.neighbor_finder = neighbor_finder