import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from optimizers.muon import Muon
from optimizers.one_sided import OneSided

# -----------------------------------------------------------------------------
# fixed config (you can still override these inside the function or via globals)
DATASET     = 'shakespeare_char'
OUT_DIR     = 'out'
BLOCK_SIZE  = 64
BATCH_SIZE  = 12
ACCUM_STEPS = 5 * 8
N_LAYER     = 4
N_HEAD      = 4
N_EMBD      = 128
DROPOUT     = 0.0
BIAS        = False
LEARNING_RATE = 6e-4
MAX_ITERS     = 350   # you can shorten for BO
EVAL_ITERS    = 350
EVAL_INTERVAL = 349
LOG_INTERVAL  = 1
DECAY_LR      = True
WARMUP_ITERS  = 450
LR_DECAY_ITERS= 450
MIN_LR        = 6e-5
GRAD_CLIP     = 1.0

learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE       = 'float16' if DEVICE=='cuda' else 'float32'
# -----------------------------------------------------------------------------

def train_with_modified_muon(muon_lr, weight_decay, momentum, cov_momentum):
    """
    Train GPT for a fixed number of iterations using ModifiedMuon with hyperparameters:
      - muon_lr
      - weight_decay
      - momentum (optimizer momentum)
      - cov_momentum (running average momentum for covariance updates)
    Returns: negative validation loss for BayesianOptimization.
    """
    # Setup device, dtype, scaler, context
    device = torch.device(DEVICE)
    ptdtype = getattr(torch, DTYPE)
    ctx = nullcontext() if DEVICE=='cpu' else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE=='float16'))

    # Data loader
    data_dir = os.path.join('data', DATASET)
    def get_batch(split):
        path = os.path.join(data_dir, f'{split}.bin')
        data = np.memmap(path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
        if DEVICE.startswith('cuda'):
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # Model init
    meta_p = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_p):
        with open(meta_p,'rb') as f:
            vocab_size = pickle.load(f)['vocab_size']
    else:
        vocab_size = 65
    cfg_model = GPTConfig(
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        block_size=BLOCK_SIZE, bias=BIAS,
        vocab_size=vocab_size, dropout=DROPOUT
    )
    model = GPT(cfg_model).to(device)
    model.train()

    # Optimizers: ModifiedMuon for weight matrices, AdamW for biases/LayerNorm
    params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    muon_params  = [p for p in params.values() if p.dim() >= 2]
    other_params = [p for p in params.values() if p.dim() < 2]

    mod_muon = OneSided(
        muon_params,
        lr=muon_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        cov_momentum=cov_momentum
    )
    adamw = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), "cpu", ignore_params=True)
    optimizers = [mod_muon, adamw]

    # LR schedule
    def get_lr(it, base_lr):
        if it < WARMUP_ITERS:
            return base_lr * (it+1)/(WARMUP_ITERS+1)
        if it > LR_DECAY_ITERS:
            return MIN_LR
        decay = 0.5*(1 + math.cos(math.pi*(it-WARMUP_ITERS)/(LR_DECAY_ITERS-WARMUP_ITERS)))
        return MIN_LR + decay*(base_lr - MIN_LR)

    # Training loop
    best_val = float('inf')
    iter_num = 0
    X, Y = get_batch('train')
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    while iter_num < MAX_ITERS:
        # Update learning rates
        lr_muon = get_lr(iter_num, muon_lr) if DECAY_LR else muon_lr
        lr_adam = get_lr(iter_num, LEARNING_RATE) if DECAY_LR else LEARNING_RATE
        for g in mod_muon.param_groups:
            g['lr'] = lr_muon
        for g in adamw.param_groups:
            g['lr'] = lr_adam

        # Forward/backward with gradient accumulation
        for _ in range(ACCUM_STEPS):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / ACCUM_STEPS
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # Gradient clipping and optimizer step
        if GRAD_CLIP > 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Logging
        if iter_num % LOG_INTERVAL == 0:
            print(f"iter {iter_num}: loss {loss.item() * ACCUM_STEPS:.4f}")

        # Evaluation
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print("HERE", losses)
            val_loss = losses['val']
            if val_loss < best_val:
                best_val = val_loss

        iter_num += 1

    # Return negative val loss for maximization
    return -best_val

