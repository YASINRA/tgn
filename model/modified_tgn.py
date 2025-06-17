# modified_tgn.py
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
from modules.tppr_embedding import TPPREmbedding, TPPRReducer

class TGN_with_TPPR(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, 
                 edge_list=None, n_layers=2, n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp", mean_time_shift_src=0, std_time_shift_src=1, 
                 mean_time_shift_dst=0, std_time_shift_dst=1, n_neighbors=None, 
                 aggregator_type="last", memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False, dyrep=False,
                 use_tppr=True, tppr_weight=0.3):
        super(TGN_with_TPPR, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.use_tppr = use_tppr
        self.tppr_weight = tppr_weight

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

        # Initialize TPPR embeddings
        if self.use_tppr and edge_list is not None:
            print("Initializing TPPR embedding...")
            self.tppr_embedding = TPPREmbedding(edge_list, embedding_dim=self.n_node_features)
            
            # Get unique nodes from edge list
            all_graph_nodes = set()
            for from_node, to_node, _ in edge_list:
                all_graph_nodes.add(from_node)
                all_graph_nodes.add(to_node)
            
            # Create mapping
            self.node_id_to_matrix_idx = {}
            matrix_nodes = self.tppr_embedding.all_nodes
            for i, node_id in enumerate(matrix_nodes):
                self.node_id_to_matrix_idx[node_id] = i
            
            self.tppr_reducer = TPPRReducer(
                input_dim=len(matrix_nodes), 
                output_dim=self.n_node_features
            ).to(device)
            
            # Pre-compute TPPR embeddings
            print("Computing TPPR embeddings...")
            with torch.no_grad():
                tppr_matrix = self.tppr_embedding.get_tppr_embeddings()
                tppr_tensor = torch.from_numpy(tppr_matrix).float().to(device)
                tppr_reduced = self.tppr_reducer(tppr_tensor)
                
                # Create full TPPR features for all nodes
                tppr_features = torch.zeros(self.n_nodes, self.n_node_features, device=device)
                for node_id, matrix_idx in self.node_id_to_matrix_idx.items():
                    if node_id < self.n_nodes:
                        tppr_features[node_id] = tppr_reduced[matrix_idx]
            
            # Register as non-trainable buffer
            self.register_buffer('tppr_features_static', tppr_features)
            print(f"TPPR embeddings computed: {tppr_features.shape}")
        else:
            self.register_buffer('tppr_features_static', None)

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

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

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                       self.n_node_features, 1)

    def get_enhanced_node_features(self, node_ids):
        """Get enhanced node features without gradient issues"""
        # Always create new tensors, never modify in-place
        original_features = self.node_raw_features[node_ids]
        
        if self.use_tppr and self.tppr_features_static is not None:
            tppr_features = self.tppr_features_static[node_ids]
            # Create completely new tensor
            enhanced_features = original_features * (1 - self.tppr_weight) + \
                              tppr_features * self.tppr_weight
            return enhanced_features
        else:
            return original_features

    def compute_temporal_embeddings_safe(self, source_nodes, destination_nodes, negative_nodes, 
                                       edge_times, edge_idxs, n_neighbors=20):
        """
        Safe version that avoids all gradient issues
        """
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        # Handle memory separately to avoid gradient issues
        memory = None
        time_diffs = None
        
        if self.use_memory:
            if self.memory_update_at_start:
                # Get updated memory without modifying original
                memory, last_update = self.get_updated_memory_safe(list(range(self.n_nodes)),
                                                                 self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            # Compute time differences
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)

        # Compute base embeddings
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                               source_nodes=nodes,
                                                               timestamps=timestamps,
                                                               n_layers=self.n_layers,
                                                               n_neighbors=n_neighbors,
                                                               time_diffs=time_diffs)

        # Apply TPPR enhancement
        if self.use_tppr and self.tppr_features_static is not None:
            enhanced_features = self.get_enhanced_node_features(nodes)
            # Create new tensor instead of modifying existing one
            node_embedding = node_embedding * (1 - self.tppr_weight) + \
                           enhanced_features * self.tppr_weight

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        # Handle memory updates safely
        if self.use_memory:
            self.update_memory_safe(positives, source_nodes, destination_nodes, 
                                  source_node_embedding, destination_node_embedding,
                                  edge_times, edge_idxs)

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, 
                                  edge_times, edge_idxs, n_neighbors=20):
        """Main interface - delegates to safe version"""
        return self.compute_temporal_embeddings_safe(source_nodes, destination_nodes, negative_nodes,
                                                    edge_times, edge_idxs, n_neighbors)

    def get_updated_memory_safe(self, nodes, messages):
        """Safe memory update that doesn't modify tensors in-place"""
        if not self.use_memory:
            return None, None
            
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
            
        # Create completely new memory tensors
        current_memory = self.memory.memory.data.clone().detach()
        current_last_update = self.memory.last_update.data.clone().detach()
        
        if len(unique_nodes) > 0:
            # Update memory for specific nodes without in-place operations
            node_memory = current_memory[unique_nodes]
            updated_node_memory = self.memory_updater.memory_updater(unique_messages, node_memory)
            
            # Create new memory tensor
            new_memory = current_memory.clone()
            new_memory[unique_nodes] = updated_node_memory
            
            new_last_update = current_last_update.clone()
            new_last_update[unique_nodes] = unique_timestamps
            
            return new_memory, new_last_update
        
        return current_memory, current_last_update

    def update_memory_safe(self, positives, source_nodes, destination_nodes,
                         source_embeddings, destination_embeddings, edge_times, edge_idxs):
        """Safe memory update without gradient issues"""
        if not self.use_memory:
            return
            
        # Clear messages first
        self.memory.clear_messages(positives)
        
        # Get new messages
        unique_sources, source_id_to_messages = self.get_raw_messages_safe(
            source_nodes, source_embeddings, destination_nodes, destination_embeddings, 
            edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.get_raw_messages_safe(
            destination_nodes, destination_embeddings, source_nodes, source_embeddings,
            edge_times, edge_idxs)
        
        # Store messages
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)

    def get_raw_messages_safe(self, source_nodes, source_embeddings, destination_nodes,
                            destination_embeddings, edge_times, edge_idxs):
        """Safe message creation without gradient issues"""
        edge_times_tensor = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        # Get memory without gradient issues
        source_memory = source_embeddings if self.use_source_embedding_in_message else \
                       self.memory.get_memory(source_nodes).detach()
        destination_memory = destination_embeddings if self.use_destination_embedding_in_message else \
                           self.memory.get_memory(destination_nodes).detach()

        source_time_delta = edge_times_tensor - self.memory.last_update[source_nodes].detach()
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

        # Create message tensor
        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                  source_time_delta_encoding], dim=1)
        
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i].detach(), edge_times_tensor[i].detach()))

        return unique_sources, messages

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                  torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, nodes, messages):
        """Legacy interface for compatibility"""
        if not self.use_memory:
            return
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                        timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        """Legacy interface for compatibility"""
        return self.get_updated_memory_safe(nodes, messages)

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
        """Legacy interface for compatibility"""
        return self.get_raw_messages_safe(source_nodes, source_node_embedding, 
                                        destination_nodes, destination_node_embedding,
                                        edge_times, edge_idxs)

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder