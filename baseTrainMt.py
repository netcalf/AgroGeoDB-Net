#!/usr/bin/env python
"""
ConvTEModel training script (self-contained & runnable)
=====================================================
• Absolute dataset path baked into --train_root default.
• Relative outputs (log, best_model.pt, *.png) saved in run_<timestamp>/.
• CLI flags: --fold, --device_ids, --epochs, --max_workers, etc.
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# --------------------------- Data ------------------------------------------------------- #

def load_data_cpu(file_path: Path, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Read one Excel file on CPU and return numpy arrays."""
    logging.info("Loading %s", file_path)
    df = pd.read_excel(file_path)
    feats = df[feat_cols].values.reshape(-1, 1, len(feat_cols))  # numpy
    tgts = df["type"].values                                     # numpy
    return feats, tgts


def to_device(feats: np.ndarray, tgts: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move numpy arrays to target device as tensors."""
    feats_t = torch.tensor(feats, dtype=torch.float32, device=device)
    tgts_t = torch.tensor(tgts, dtype=torch.long, device=device)
    return feats_t, tgts_t

# --------------------------- Model ------------------------------------------------------ #

class VAE(nn.Module):
    def __init__(self, in_dim: int = 14, latent: int = 14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
            nn.ReLU(),
        )
        self.mu = nn.Linear(14, latent)
        self.logvar = nn.Linear(14, latent)

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * torch.clamp(logvar, -2, 2))
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        h = self.encoder(x.view(x.size(0), -1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = self._reparam(mu, logvar)
        return z, mu, logvar


class ConvTEModel(nn.Module):
    def __init__(self, num_classes: int = 2, in_dim: int = 14, latent: int = 14, hidden: int = 14):
        super().__init__()
        self.vae = VAE(in_dim, latent)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=in_dim if i == 0 else hidden * 2,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            for i in range(2)
        ])
        self.res_fc = nn.Linear(in_dim, hidden * 2)
        self.cls = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        # x: (N, 1, C)  -> mean over dim=1 -> (N, C)
        x = x.mean(dim=1)
        x, _, _ = self.vae(x)
        x = x.unsqueeze(1).permute(1, 0, 2)  # (1, N, C)
        res = self.res_fc(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = x + res
        x = x.mean(dim=0)
        return torch.sigmoid(self.cls(x))

# --------------------------- Loss ------------------------------------------------------- #

class FocalLossReg(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reg: float = 1e-5):
        super().__init__()
        self.alpha, self.gamma, self.reg = alpha, gamma, reg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):  # type: ignore[override]
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        reg_term = sum(p.pow(2).sum() for p in self.parameters())
        return focal.mean() + self.reg * reg_term

# --------------------------- Train / Eval ---------------------------------------------- #

def train_epoch(model, data, crit, opt, device):
    model.train()
    total = 0.0
    for feats, tgts in data:
        opt.zero_grad()
        out = model(feats.to(device))
        loss = crit(out, tgts.to(device))
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(data)


def eval_epoch(model, data, crit, device):
    model.eval()
    mets = {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
    with torch.no_grad():
        for feats, tgts in data:
            out = model(feats.to(device))
            loss = crit(out, tgts.to(device)).item()
            preds = out.argmax(1).cpu().numpy()
            labels = tgts.cpu().numpy()
            mets['loss'].append(loss)
            mets['acc'].append(accuracy_score(labels, preds))
            mets['prec'].append(precision_score(labels, preds, average='macro', zero_division=0))
            mets['rec'].append(recall_score(labels, preds, average='macro', zero_division=0))
            mets['f1'].append(f1_score(labels, preds, average='macro', zero_division=0))
    return {k: float(np.mean(v)) for k, v in mets.items()}

# --------------------------- Utils ------------------------------------------------------ #

DEFAULT_ROOT = "/home/ubuntu/Data/hyw/tractor/calculateData6(10-e=0.08,m=2,d=0.08,i=0.002)/tractor/_10fold"
FEATURE_COLS = [
    "distance", "speed", "speedDiff", "acceleration", "bearing", "bearingDiff",
    "bearingSpeed", "bearingSpeedDiff", "curvature", "distance_five", "distance_ten",
    "distribution", "angle_std", "angle_mean"
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--train_root', default=DEFAULT_ROOT)
    p.add_argument('--fold', type=int, default=9)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--device_ids', type=int, nargs='+', default=[0, 1])
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--max_workers', type=int, default=16,
                   help='number of worker processes for loading Excel files')
    return p.parse_args()


def setup_logging(rundir: Path):
    rundir.mkdir(parents=True, exist_ok=True)
    log_file = rundir / f'train_{rundir.name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    script_name = Path(sys.argv[0]).name
    logging.info('script: %s | Log file: %s', script_name, log_file)


def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_plots(rundir: Path, epochs: List[int], hist: Dict[str, List[float]]):
    pairs = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_acc', 'Validation Accuracy'),
        ('val_prec', 'Validation Precision'),
        ('val_rec', 'Validation Recall'),
        ('val_f1', 'Validation F1'),
    ]
    for key, title in pairs:
        plt.figure()
        plt.plot(epochs, hist[key], label=title)
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(rundir / f'{key}.png')
        plt.close()

# --------------------------- main ------------------------------------------------------- #

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rundir = Path.cwd() / f'run_{timestamp}'
    setup_logging(rundir)
    set_seed()

    device = torch.device(f'cuda:{args.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device(s): %s', args.device_ids)

    # ---------------- Data ---------------- #
    fold_path = Path(args.train_root) / str(args.fold)
    train_files = list((fold_path / 'train').glob('*.xlsx'))
    val_files = list((fold_path / 'valid').glob('*.xlsx'))

    if not train_files or not val_files:
        logging.error('No data found under %s', fold_path)
        sys.exit(1)

    # 并行读 CPU 数据
    cpu_train: List[Tuple[np.ndarray, np.ndarray]] = []
    cpu_val: List[Tuple[np.ndarray, np.ndarray]] = []

    logging.info("Loading data with %d workers ...", args.max_workers)
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        # 提交所有 train 任务
        fut2kind = {ex.submit(load_data_cpu, fp, FEATURE_COLS): ('train', fp) for fp in train_files}
        # 提交所有 val 任务
        fut2kind.update({ex.submit(load_data_cpu, fp, FEATURE_COLS): ('val', fp) for fp in val_files})

        for fut in as_completed(fut2kind):
            kind, fp = fut2kind[fut]
            feats, tgts = fut.result()
            if kind == 'train':
                cpu_train.append((feats, tgts))
            else:
                cpu_val.append((feats, tgts))

    # 统一搬到 GPU
    train_data = [to_device(feats, tgts, device) for feats, tgts in cpu_train]
    val_data = [to_device(feats, tgts, device) for feats, tgts in cpu_val]

    # ---------------- Model ---------------- #
    model = ConvTEModel().to(device)
    if len(args.device_ids) > 1 and torch.cuda.device_count() >= len(args.device_ids):
        model = nn.DataParallel(model, device_ids=args.device_ids)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLossReg()

    best_mix = -math.inf
    patience_ctr = 0
    history = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1']}

    MIX = lambda met: 0.2 * met['prec'] + 0.2 * met['rec'] + 0.3 * met['f1'] + 0.3 * met['acc']

    for epoch in range(1, args.epochs + 1):
        tloss = train_epoch(model, train_data, criterion, optimizer, device)

        # 只在每10个 epoch 做一次验证，模仿原脚本
        if epoch % 10 == 0 or epoch == args.epochs:
            val_data_shuffled = val_data[:]  # 浅拷贝
            random.shuffle(val_data_shuffled)
            met = eval_epoch(model, val_data_shuffled, criterion, device)
            mix_score = MIX(met)
            logging.info(
                'Epoch %d/%d | TL %.4f | VL %.4f | ACC %.4f | PREC %.4f | REC %.4f | F1 %.4f | MIX %.4f',
                epoch, args.epochs, tloss, met['loss'], met['acc'], met['prec'], met['rec'], met['f1'], mix_score
            )

            history['train_loss'].append(tloss)
            history['val_loss'].append(met['loss'])
            history['val_acc'].append(met['acc'])
            history['val_prec'].append(met['prec'])
            history['val_rec'].append(met['rec'])
            history['val_f1'].append(met['f1'])

            if mix_score > best_mix:
                best_mix = mix_score
                patience_ctr = 0
                to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(to_save, rundir / 'best_model.pt')
                logging.info(' ✓ New best model saved (MIX=%.4f)', best_mix)
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    logging.info('Early stopping: no MIX improvement for %d evals', patience_ctr)
                    break
        else:
            logging.info('Epoch %d/%d | TL %.4f (no validation this epoch)', epoch, args.epochs, tloss)

    # plots
    epochs_axis = list(range(10, 10 * len(history['val_loss']) + 1, 10))
    save_plots(rundir, epochs_axis, history)
    logging.info('Training completed. Outputs in %s', rundir)


if __name__ == '__main__':
    main()
