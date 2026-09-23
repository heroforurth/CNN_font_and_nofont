"""Save a grid of augmented samples so you can eyeball the augmentation strength.

Usage: python preview_augment.py [dataset_dir] [out_file]
Each row = one random image: original (left) followed by N augmented versions.
"""
import random
import sys

import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from augment import train_transform, test_transform, MEAN, STD

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else 'dataset/round2'
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else 'augment_preview.png'
ROWS, AUG_PER_ROW = 8, 7

mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)

dataset = ImageFolder(root=DATA_DIR)
tiles = []
for idx in random.sample(range(len(dataset)), ROWS):
    img, _ = dataset[idx]
    img = img.convert('RGB')
    tiles.append(test_transform(img) * std + mean)
    tiles += [train_transform(img) * std + mean for _ in range(AUG_PER_ROW)]

save_image(torch.stack(tiles).clamp(0, 1), OUT_FILE, nrow=AUG_PER_ROW + 1, padding=4, pad_value=0.5)
print(f"Saved {OUT_FILE}")
