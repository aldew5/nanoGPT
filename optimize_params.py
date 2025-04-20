import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from bayes_opt import BayesianOptimization
from train_func import *


# Define parameter bounds for optimization
pbounds = {
    'muon_lr': (1e-4, 1e-2),  # Learning rate for ModifiedMuon
    'weight_decay': (0, 1e-2),  # Weight decay
    'momentum': (0.8, 0.99),  # Momentum
    'cov_momentum': (0.8, 0.99),
}

# Initialize Bayesian optimizer
optimizer = BayesianOptimization(
    f=train_with_modified_muon,
    pbounds=pbounds,
    random_state=1,
)

# Run optimization
optimizer.maximize(
    init_points=5,
    n_iter=25
)

# Print best parameters
print("Best parameters found:")
print(optimizer.max)
