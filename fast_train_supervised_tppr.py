import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from model.modified_tgn import TGN_with_TPPR
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Disable anomaly detection for speed
torch.autograd.set_detect_anomaly(False)

### Argument and global variables
parser = argparse.ArgumentParser('Fast TGN supervised training with TPPR')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')  # Increased batch size
parser.add_argument('--prefix', type=str, default='tppr-fast', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--tppr_weight', type=float, default=0.3, help='Weight for TPPR features')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='Path to pretrained self-supervised model')
parser.add_argument('--disable_memory_update', action='store_true',
                    help='Disable memory updates during supervised training for speed')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}-node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}-{}.log'.format(args.prefix, str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# Load data
full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)
train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Prepare edge list for TPPR (only if not loading pretrained)
edge_list = None
if not args.pretrained_model:
    def prepare_edge_list_for_tppr(data):
        edge_list = []
        for i in range(len(data.sources)):
            edge_list.append((data.sources[i], data.destinations[i], data.timestamps[i]))
        return edge_list
    edge_list = prepare_edge_list_for_tppr(train_data)

for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix, i) if i > 0 else \
                 "results/{}_node_classification.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  if args.pretrained_model:
    logger.info(f'Loading pretrained model from {args.pretrained_model}')
    # Load pretrained TGN and extract TPPR features
    pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)
    
    # Initialize TGN with TPPR features from pretrained model
    tgn = TGN_with_TPPR(neighbor_finder=train_ngh_finder, 
                        node_features=node_features,
                        edge_features=edge_features, 
                        device=device,
                        edge_list=None,  # Skip TPPR computation
                        n_layers=NUM_LAYER,
                        n_heads=NUM_HEADS, 
                        dropout=DROP_OUT, 
                        use_memory=USE_MEMORY,
                        message_dimension=MESSAGE_DIM, 
                        memory_dimension=MEMORY_DIM,
                        memory_update_at_start=not args.memory_update_at_end,
                        embedding_module_type=args.embedding_module,
                        message_function=args.message_function,
                        aggregator_type=args.aggregator, 
                        n_neighbors=NUM_NEIGHBORS,
                        mean_time_shift_src=mean_time_shift_src, 
                        std_time_shift_src=std_time_shift_src,
                        mean_time_shift_dst=mean_time_shift_dst, 
                        std_time_shift_dst=std_time_shift_dst,
                        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                        use_source_embedding_in_message=args.use_source_embedding_in_message,
                        use_tppr=True,
                        tppr_weight=args.tppr_weight)
    
    # Load pretrained weights
    tgn.load_state_dict(pretrained_checkpoint, strict=False)
    logger.info('Pretrained model loaded successfully')
    
  else:
    # Initialize new model with TPPR
    tgn = TGN_with_TPPR(neighbor_finder=train_ngh_finder, 
                        node_features=node_features,
                        edge_features=edge_features, 
                        device=device,
                        edge_list=edge_list,
                        n_layers=NUM_LAYER,
                        n_heads=NUM_HEADS, 
                        dropout=DROP_OUT, 
                        use_memory=USE_MEMORY,
                        message_dimension=MESSAGE_DIM, 
                        memory_dimension=MEMORY_DIM,
                        memory_update_at_start=not args.memory_update_at_end,
                        embedding_module_type=args.embedding_module,
                        message_function=args.message_function,
                        aggregator_type=args.aggregator, 
                        n_neighbors=NUM_NEIGHBORS,
                        mean_time_shift_src=mean_time_shift_src, 
                        std_time_shift_src=std_time_shift_src,
                        mean_time_shift_dst=mean_time_shift_dst, 
                        std_time_shift_dst=std_time_shift_dst,
                        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                        use_source_embedding_in_message=args.use_source_embedding_in_message,
                        use_tppr=True,
                        tppr_weight=args.tppr_weight)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  
  logger.info('Num of training instances: {}'.format(num_instance))
  logger.info('Num of batches per epoch: {}'.format(num_batch))

  # Freeze TGN parameters, only train classifier
  for name, param in tgn.named_parameters():
    param.requires_grad = False
  
  # Only enable gradients for TPPR reducer if it exists
  if hasattr(tgn, 'tppr_reducer') and tgn.tppr_reducer is not None:
    for param in tgn.tppr_reducer.parameters():
      param.requires_grad = True
    logger.info('TPPR reducer parameters enabled for training')
  
  logger.info('TGN parameters frozen for fast training')

  # Initialize classifier
  decoder = MLP(tgn.n_node_features, drop=DROP_OUT)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCELoss()

  # Optimizer only for trainable parameters
  trainable_params = list(decoder.parameters())
  if hasattr(tgn, 'tppr_reducer') and tgn.tppr_reducer is not None:
    trainable_params.extend(list(tgn.tppr_reducer.parameters()))
  
  optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
  
  logger.info(f'Total trainable parameters: {sum(p.numel() for p in trainable_params)}')

  val_aucs = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    
    # Initialize memory once per epoch
    if USE_MEMORY:
      with torch.no_grad():
        tgn.memory.__init_memory__()

    # Set to eval mode to freeze TGN, train mode for classifier
    tgn.eval()
    if hasattr(tgn, 'tppr_reducer') and tgn.tppr_reducer is not None:
      tgn.tppr_reducer.train()
    decoder.train()
    
    epoch_loss = 0
    
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      optimizer.zero_grad()
      
      # Fast forward pass with minimal gradient computation
      if args.disable_memory_update or not USE_MEMORY:
        # Completely disable memory updates for speed
        with torch.no_grad():
          # Get base embeddings without memory updates
          node_embedding = tgn.embedding_module.compute_embedding(
              memory=None,
              source_nodes=sources_batch,
              timestamps=timestamps_batch,
              n_layers=tgn.n_layers,
              n_neighbors=NUM_NEIGHBORS,
              time_diffs=None)
          
          # Apply TPPR enhancement
          if tgn.use_tppr and tgn.tppr_features_static is not None:
            enhanced_features = tgn.get_enhanced_node_features(sources_batch)
            source_embedding = node_embedding * (1 - tgn.tppr_weight) + \
                             enhanced_features * tgn.tppr_weight
          else:
            source_embedding = node_embedding
      else:
        # Use memory but freeze most computations
        with torch.no_grad():
          # Get memory without updates
          if tgn.memory_update_at_start:
            memory = tgn.memory.get_memory(list(range(tgn.n_nodes)))
            last_update = tgn.memory.last_update
          else:
            memory = tgn.memory.get_memory(list(range(tgn.n_nodes)))
            last_update = tgn.memory.last_update
          
          # Compute base embeddings
          node_embedding = tgn.embedding_module.compute_embedding(
              memory=memory,
              source_nodes=sources_batch,
              timestamps=timestamps_batch,
              n_layers=tgn.n_layers,
              n_neighbors=NUM_NEIGHBORS,
              time_diffs=None)
        
        # Apply TPPR enhancement (with gradients if TPPR reducer is trainable)
        if tgn.use_tppr and tgn.tppr_features_static is not None:
          enhanced_features = tgn.get_enhanced_node_features(sources_batch)
          source_embedding = node_embedding * (1 - tgn.tppr_weight) + \
                           enhanced_features * tgn.tppr_weight
        else:
          source_embedding = node_embedding

      # Classification with gradients
      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      pred = decoder(source_embedding).sigmoid()
      loss = decoder_loss_criterion(pred, labels_batch_torch)
      
      # Backward pass
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item()
      
    train_losses.append(epoch_loss / num_batch)

    # Fast validation
    tgn.eval()
    decoder.eval()
    
    with torch.no_grad():
      val_auc = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                         n_neighbors=NUM_NEIGHBORS)
    val_aucs.append(val_auc)

    # Save progress
    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": [time.time() - start_epoch],
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    logger.info(f'Epoch {epoch}: train loss: {epoch_loss / num_batch:.4f}, val auc: {val_auc:.4f}, time: {time.time() - start_epoch:.2f}s')
    
    # Early stopping
    if args.use_validation:
      if early_stopper.early_stop_check(val_auc):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        break

  # Final test
  tgn.eval()
  decoder.eval()
  
  with torch.no_grad():
    test_auc = eval_node_classification(tgn, decoder, test_data, full_data.edge_idxs, BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS)
    
  logger.info(f'Final test auc: {test_auc:.4f}')

  # Save final model
  torch.save({
    'tgn_state_dict': tgn.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'args': args,
    'test_auc': test_auc,
    'val_aucs': val_aucs,
    'train_losses': train_losses
  }, MODEL_SAVE_PATH)
  logger.info(f'Model saved to {MODEL_SAVE_PATH}')