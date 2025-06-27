import numpy as np
from QTCS.qtcs import Graph

DATASET_NAME = "wikipedia"  # Change if needed
ALPHA = 0.2
TOPK = 128

csv_path = f"data/graph.txt"
tppr_out_path = f"data/tppr_embeddings.npy"
tppr_topk_values_path = f"data/tppr_topk_values.npy"

print(f"Loading temporal graph from {csv_path} ...")
G_qtcs = Graph(csv_path)

num_nodes = max(G_qtcs.tadj_list.keys()) + 1
print(f"Computing TPPR for {num_nodes} nodes ...")
tppr_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for node_id in range(num_nodes):
    tppr, _ = G_qtcs.Compute_tppr(ALPHA, node_id)
    for target_id, score in tppr.items():
        tppr_matrix[node_id, target_id] = score
    if node_id % 100 == 0:
        print(f"  Done {node_id}/{num_nodes}")

np.save(tppr_out_path, tppr_matrix)
print(f"Saved full TPPR matrix: {tppr_matrix.shape} -> {tppr_out_path}")

# Dimensionality reduction: keep only top-k TPPR values per node
print(f"Reducing TPPR to top-{TOPK} values per node ...")
def reduce_tppr_topk(tppr_matrix, k=TOPK):
    num_nodes = tppr_matrix.shape[0]
    topk_values = np.zeros((num_nodes, k), dtype=np.float32)
    for i in range(num_nodes):
        idx = np.argpartition(-tppr_matrix[i], k)[:k]
        vals = tppr_matrix[i, idx]
        order = np.argsort(-vals)
        topk_values[i] = vals[order]
    return topk_values

tppr_topk = reduce_tppr_topk(tppr_matrix, k=TOPK)
np.save(tppr_topk_values_path, tppr_topk)
print(f"Saved reduced TPPR (top-{TOPK}) embedding: {tppr_topk.shape} -> {tppr_topk_values_path}")