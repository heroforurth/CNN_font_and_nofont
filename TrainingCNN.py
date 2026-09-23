"""Fine-tune ResNet-18 on Thai characters.

Data
  - dataset/round2            real images, stratified 80/20 train/val split
  - dataset/synth_fonts/train font-rendered images (render_fonts.py), added to training
  - dataset/synth_fonts/val   held-out font families, second validation set
  - archive/synthetic_test_set  TEST set: only evaluated once at the very end,
                                never used for model selection

Model selection: mean of (real val acc, font val acc), best epoch weights are always restored.

--no-synth: real data only (no font images anywhere). The 20% real split becomes a pure test set:
            train the full schedule, keep the final epoch, evaluate the 20% once at the end.
"""
import argparse
import csv
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

# Augmentations live in augment.py (run preview_augment.py to see them)
from augment import train_transform, test_transform

# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------
REAL_DIR = 'dataset/round2'
SYNTH_TRAIN_DIR = 'dataset/synth_fonts/train'
SYNTH_VAL_DIR = 'dataset/synth_fonts/val'
TEST_DIR = 'archive/synthetic_test_set'
USE_SYNTH = True

NUM_CLASSES = 72
SEED = 42
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
PATIENCE = EPOCHS        # OneCycle does its best work at the end of the schedule; best-val epoch is still kept
NUM_WORKERS = 6       # train loader; each Windows worker commits ~1.5 GB (torch DLLs), keep this modest
VAL_WORKERS = 1
OUT_PATH = 'model.pt'
LOG_PATH = 'train_log.csv'


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
class ImageListDataset(Dataset):
    """(path, label) list -> transformed RGB tensor."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
        return self.transform(img), label

    def __len__(self):
        return len(self.samples)


def stratified_split(samples, val_frac, seed):
    """Split per class so rare classes appear in both train and val (when they have >=2 images)."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n_val = min(int(round(len(items) * val_frac)), len(items) - 1)
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def folder_samples(root, class_to_idx):
    """Load an ImageFolder-style dir using the REAL dataset's class_to_idx."""
    ds = ImageFolder(root)
    return [(p, class_to_idx[ds.classes[y]]) for p, y in ds.samples]


def make_loader(samples, transform, workers=VAL_WORKERS, **kw):
    return DataLoader(ImageListDataset(samples, transform), batch_size=BATCH_SIZE,
                      num_workers=workers, pin_memory=True, persistent_workers=True, **kw)


# --------------------------------------------------------------------------------------------
# Model / eval
# --------------------------------------------------------------------------------------------
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(  # type: ignore
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy %, per-sample predictions)."""
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0
    preds = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            outputs = model(images)
        loss_sum += criterion(outputs.float(), labels) * labels.size(0)
        pred = outputs.argmax(1)
        correct += pred.eq(labels).sum()
        total += labels.size(0)
        preds.append(pred.cpu())
    return loss_sum.item() / total, correct.item() / total * 100, torch.cat(preds)


def main():
    global USE_SYNTH, OUT_PATH, LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-synth', action='store_true', help='real data only; 20%% split = pure test set')
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--log', default=LOG_PATH)
    args = ap.parse_args()
    USE_SYNTH, OUT_PATH, LOG_PATH = not args.no_synth, args.out, args.log

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    # ---- datasets ----
    real = ImageFolder(REAL_DIR)
    class_to_idx = real.class_to_idx
    assert len(class_to_idx) == NUM_CLASSES
    train_samples, real_val_samples = stratified_split(real.samples, 0.2, SEED)
    n_real_train = len(train_samples)
    if USE_SYNTH:
        train_samples += folder_samples(SYNTH_TRAIN_DIR, class_to_idx)
    font_val_samples = folder_samples(SYNTH_VAL_DIR, class_to_idx) if USE_SYNTH else []
    test_ds = ImageFolder(TEST_DIR)
    test_samples = [(p, class_to_idx[test_ds.classes[y]]) for p, y in test_ds.samples]

    # class balancing over the combined training set
    targets = np.array([y for _, y in train_samples])
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    sample_weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(targets), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = make_loader(train_samples, train_transform, workers=NUM_WORKERS, sampler=sampler,
                               drop_last=True, prefetch_factor=4)
    if USE_SYNTH:
        real_val_loader = make_loader(real_val_samples, test_transform)
        font_val_loader = make_loader(font_val_samples, test_transform)

    print(f"Train: {len(train_samples)} ({n_real_train} real + {len(train_samples) - n_real_train} synth) | "
          f"Real {'val' if USE_SYNTH else 'TEST (20%)'}: {len(real_val_samples)} | "
          f"Font val: {len(font_val_samples)} | Font test: {len(test_samples)}")
    print(f"Smallest classes in train: {sorted(class_counts)[:5]}")

    # ---- model / optim ----
    model = build_model().to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS,
                                                    steps_per_epoch=len(train_loader), pct_start=0.15)

    best_score, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'real_val_loss', 'real_val_acc',
                                'font_val_loss', 'font_val_acc', 'score', 'seconds'])

    print(f"Starting training on {device}...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                outputs = model(images)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # keep sums on GPU to avoid a CPU sync every batch
            running_loss += loss.detach() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum()
            total += labels.size(0)

        train_loss = running_loss.item() / total
        train_acc = correct.item() / total * 100
        if not USE_SYNTH:
            # nothing is evaluated during training: the 20% split stays untouched until the end
            secs = time.time() - t0
            print(f"Epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} acc {train_acc:.2f}% | {secs:.0f}s", flush=True)
            with open(LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", '', '', '', '', '', f"{secs:.0f}"])
            best_epoch, best_state = epoch, None
            continue
        rv_loss, rv_acc, _ = evaluate(model, real_val_loader, criterion, device)
        fv_loss, fv_acc, _ = evaluate(model, font_val_loader, criterion, device)
        score = (rv_acc + fv_acc) / 2
        secs = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"real val {rv_acc:.2f}% | font val {fv_acc:.2f}% | score {score:.2f} | {secs:.0f}s", flush=True)
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{rv_loss:.4f}", f"{rv_acc:.2f}",
                                    f"{fv_loss:.4f}", f"{fv_acc:.2f}", f"{score:.2f}", f"{secs:.0f}"])

        if score > best_score + 0.01:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[Early Stopping] no improvement for {PATIENCE} epochs")
                break

    # ---- restore best & final test (only place the test set is used) ----
    if USE_SYNTH:
        model.load_state_dict(best_state)
        print(f"\nBest epoch {best_epoch} (val score {best_score:.2f})")
    else:
        print(f"\nUsing final epoch {best_epoch}")
    _, real_acc, _ = evaluate(model, make_loader(real_val_samples, test_transform), criterion, device)
    print(f"Real 20% split accuracy: {real_acc:.2f}%")
    test_loader = make_loader(test_samples, test_transform)
    _, test_acc, test_preds = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # per-font breakdown (test filenames look like synth_161_<font>.png)
    per_font = defaultdict(lambda: [0, 0])
    for (path, y), p in zip(test_samples, test_preds.tolist()):
        font = os.path.splitext(os.path.basename(path))[0].split('_', 2)[2]
        per_font[font][0] += int(p == y)
        per_font[font][1] += 1
    for font, (c, n) in sorted(per_font.items()):
        print(f"  {font:20s} {c / n * 100:6.2f}% ({c}/{n})")

    torch.save({'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                'class_to_idx': class_to_idx}, OUT_PATH)
    print(f"Model saved as {OUT_PATH}")


if __name__ == '__main__':
    main()
