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

from Modified_TGN.modified_tgn import TGN_with_TPPR
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Disable anomaly detection for better performance
torch.autograd.set_detect_anomaly(False)

### Argument and global variables
parser = argparse.ArgumentParser('TGN supervised training with TPPR')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tppr', help='Prefix to name the checkpoints')
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
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
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
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--tppr_weight', type=float, default=0.3, help='Weight for TPPR features')
parser.add_argument('--training_mode', type=str, default='full', choices=['full', 'freeze_tgn', 'tppr_only'],
                    help='Training mode: full (train all), freeze_tgn (freeze TGN), tppr_only (only TPPR)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}-node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}-node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
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

# Prepare edge list for TPPR
def prepare_edge_list_for_tppr(data):
    """Prepare edge list format for TPPR"""
    edge_list = []
    for i in range(len(data.sources)):
        edge_list.append((
            data.sources[i],
            data.destinations[i], 
            data.timestamps[i]
        ))
    return edge_list

edge_list = prepare_edge_list_for_tppr(train_data)

for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                i) if i > 0 else "results/{}_node_classification.pkl".format(
    args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model with TPPR
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
  logger.info(f'Training mode: {args.training_mode}')

  # Configure training mode
  if args.training_mode == 'freeze_tgn':
    # Freeze all TGN parameters except TPPR reducer
    for name, param in tgn.named_parameters():
      if 'tppr_reducer' not in name:
        param.requires_grad = False
    logger.info('TGN parameters frozen, only TPPR reducer trainable')
  elif args.training_mode == 'tppr_only':
    # Only train TPPR reducer
    for name, param in tgn.named_parameters():
      if 'tppr_reducer' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
    logger.info('Only TPPR reducer trainable')
  else:
    # Train everything
    logger.info('Training all parameters')

  # Initialize classifier
  decoder = MLP(tgn.n_node_features, drop=DROP_OUT)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCELoss()

  # Setup optimizer with only trainable parameters
  trainable_params = []
  
  # Add decoder parameters
  for param in decoder.parameters():
    trainable_params.append(param)
  
  # Add TGN parameters that require gradients
  for name, param in tgn.named_parameters():
    if param.requires_grad:
      trainable_params.append(param)
      logger.info(f'Training parameter: {name}')
  
  decoder_optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
  logger.info(f'Total trainable parameters: {len(trainable_params)}')

  val_aucs = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    
    # Initialize memory at start of each epoch
    if USE_MEMORY:
      with torch.no_grad():
        tgn.memory.__init_memory__()

    # Set training modes
    if args.training_mode in ['freeze_tgn', 'tppr_only']:
      tgn.eval()
      # Only set TPPR reducer to train mode
      if hasattr(tgn, 'tppr_reducer'):
        tgn.tppr_reducer.train()
    else:
      tgn.train()
    
    decoder.train()
    
    total_loss = 0
    
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      # Clear gradients
      decoder_optimizer.zero_grad()
      
      try:
        # Compute embeddings
        if args.training_mode in ['freeze_tgn', 'tppr_only']:
          # Use no_grad for most of TGN computation
          with torch.no_grad():
            # Get base embeddings without gradients
            source_embedding_base, _, _ = tgn.compute_temporal_embeddings(
                sources_batch,
                destinations_batch,
                destinations_batch,
                timestamps_batch,
                edge_idxs_batch,
                NUM_NEIGHBORS)
          
          # Apply TPPR enhancement with gradients if needed
          if args.training_mode == 'tppr_only' and tgn.use_tppr:
            # Only the TPPR part has gradients
            enhanced_features = tgn.get_enhanced_node_features(sources_batch)
            source_embedding = source_embedding_base * (1 - tgn.tppr_weight) + \
                             enhanced_features * tgn.tppr_weight
          else:
            source_embedding = source_embedding_base
        else:
          # Full training with gradients
          source_embedding, _, _ = tgn.compute_temporal_embeddings(
              sources_batch,
              destinations_batch,
              destinations_batch,
              timestamps_batch,
              edge_idxs_batch,
              NUM_NEIGHBORS)

        # Classification
        labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
        pred = decoder(source_embedding).sigmoid()
        decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
        
        # Backward pass
        decoder_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        # Update parameters
        decoder_optimizer.step()
        
        total_loss += decoder_loss.item()
        
      except RuntimeError as e:
        logger.error(f"Error in batch {k}: {e}")
        # Skip this batch and continue
        continue
      
    train_losses.append(total_loss / num_batch)

    # Validation
    tgn.eval()
    decoder.eval()
    
    with torch.no_grad():
      val_auc = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                         n_neighbors=NUM_NEIGHBORS)
    val_aucs.append(val_auc)

    # Save results
    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": [0.0],
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    logger.info(f'Epoch {epoch}: train loss: {total_loss / num_batch:.4f}, val auc: {val_auc:.4f}, time: {time.time() - start_epoch:.2f}s')
    
    # Early stopping
    if args.use_validation:
      if early_stopper.early_stop_check(val_auc):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        break
      else:
        torch.save({
          'decoder_state_dict': decoder.state_dict(),
          'tgn_state_dict': tgn.state_dict(),
          'epoch': epoch,
          'val_auc': val_auc
        }, get_checkpoint_path(epoch))

  # Final evaluation
  if args.use_validation:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_checkpoint = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    decoder.load_state_dict(best_checkpoint['decoder_state_dict'])
    tgn.load_state_dict(best_checkpoint['tgn_state_dict'])
    
    tgn.eval()
    decoder.eval()
    
    with torch.no_grad():
      test_auc = eval_node_classification(tgn, decoder, test_data, full_data.edge_idxs, BATCH_SIZE,
                                          n_neighbors=NUM_NEIGHBORS)
  else:
    test_auc = val_aucs[-1]
    
  # Save final results
  pickle.dump({
    "val_aps": val_aucs,
    "test_ap": test_auc,
    "train_losses": train_losses,
    "epoch_times": [0.0],
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
  }, open(results_path, "wb"))

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