# tppr_embedding.py
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import time

class TPPREmbedding:
    def __init__(self, edge_list, restart_prob=0.4, max_steps=1000, embedding_dim=64):
        self.graph = self.parse_edge_list(edge_list)
        self.restart_prob = restart_prob
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        self.all_nodes = set()
        for from_node, to_node, _ in edge_list:
            self.all_nodes.add(from_node)
            self.all_nodes.add(to_node)
        self.all_nodes = sorted(list(self.all_nodes))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.all_nodes)}
        
    def parse_edge_list(self, edge_list):
        graph = defaultdict(list)
        for from_node, to_node, time in edge_list:
            graph[from_node].append((to_node, time))
        return graph
    
    def random_walk_with_restart(self, start_node):
        current_node = start_node
        current_time = -1
        visit_counts = defaultdict(float)
        visit_counts[current_node] += 1.0
        
        for _ in range(self.max_steps):
            if random.random() < self.restart_prob:
                current_node = start_node
                current_time = 0
            else:
                possible_next = [(to_node, time, time-current_time) 
                               for to_node, time in self.graph[current_node] 
                               if time > current_time]
                
                if not possible_next:
                    current_node = start_node
                    current_time = 0
                else:
                    sum_of_times = sum(1/max(dis_time, 1e-6) for _, _, dis_time in possible_next)
                    if sum_of_times > 0:
                        weights = [(1/max(dis_time, 1e-6))/sum_of_times for _, _, dis_time in possible_next]
                        idx = np.random.choice(len(possible_next), p=weights)
                        next_node, next_time, next_dis_time = possible_next[idx]
                    else:
                        next_node, next_time, next_dis_time = random.choice(possible_next)
                    
                    current_node = next_node
                    current_time = next_time
                    visit_counts[current_node] += 1.0
        
        return visit_counts
    
    def compute_tppr_for_all_nodes(self):
        """Compute TPPR values for all nodes"""
        print(f"Computing TPPR for {len(self.all_nodes)} nodes...")
        tppr_matrix = np.zeros((len(self.all_nodes), len(self.all_nodes)))
        
        for i, start_node in enumerate(self.all_nodes):
            if i % 100 == 0:
                print(f"Processing node {i}/{len(self.all_nodes)}")
            
            visit_counts = self.random_walk_with_restart(start_node)
            
            # Normalize
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for node, count in visit_counts.items():
                    if node in self.node_to_idx:
                        j = self.node_to_idx[node]
                        tppr_matrix[i][j] = count / total_visits
        
        return tppr_matrix
    
    def get_tppr_embeddings(self):
        """Get TPPR embedding matrix - returns numpy array without gradients"""
        tppr_matrix = self.compute_tppr_for_all_nodes()
        return tppr_matrix  # Return numpy array, not torch tensor

class TPPRReducer(nn.Module):
    """Use TSN to reduce TPPR embedding dimensions"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(TPPRReducer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple reduction network
        self.reducer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, tppr_embeddings):
        """
        Args:
            tppr_embeddings: [num_nodes, num_nodes] TPPR matrix
        Returns:
            reduced_embeddings: [num_nodes, output_dim] reduced embeddings
        """
        reduced_embeddings = self.reducer(tppr_embeddings)
        return reduced_embeddings