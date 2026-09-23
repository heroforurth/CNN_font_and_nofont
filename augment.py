"""Data augmentation for Thai character images (black strokes on white background).

Pipeline order matters:
  1. PIL stage at NATIVE resolution (~10-35 px): stroke thickness, low-res / JPEG artifacts
  2. Resize to 224
  3. Geometric distortion at 224: affine (rotate/shift/scale/shear), perspective, elastic warp
  4. Photometric: blur, brightness/contrast
  5. Tensor stage: gaussian noise, salt & pepper, cutout, then normalize

All geometric fills are WHITE (255) so rotated/sheared corners look like background,
not black wedges the model could learn as a shortcut.
"""
import io
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WHITE = (255, 255, 255)

cv2.setNumThreads(0)  # DataLoader already parallelizes across workers


# --------------------------------------------------------------------------------------------
# PIL-stage transforms (applied at native resolution, before Resize)
# --------------------------------------------------------------------------------------------
class RandomStrokeThickness:
    """Randomly erode/dilate strokes to simulate thin or bold characters.
    Runs before Resize so a 2-3 px kernel actually changes the stroke width."""
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        # 3x3 is too destructive on very small glyphs, so only allow it on larger ones
        kernel_size = 3 if min(img.size) >= 30 and random.random() < 0.5 else 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            # dark strokes on white: erode grows the dark region -> bolder
            img_np = cv2.erode(img_np, kernel, iterations=1)
        else:
            # dilate shrinks the dark region -> thinner
            thinned = cv2.dilate(img_np, kernel, iterations=1)
            # already-thin strokes can vanish entirely; skip if over half the ink is lost
            if (thinned < 128).sum() < 0.5 * (img_np < 128).sum():
                return img
            img_np = thinned
        return Image.fromarray(img_np)


class RandomLowResolution:
    """Downscale then upscale back to simulate blurry / pixelated low-res scans."""
    def __init__(self, p=0.25, scale=(0.6, 0.85)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale)
        small = img.resize((max(4, int(w * s)), max(4, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), random.choice([Image.NEAREST, Image.BILINEAR]))


class RandomJPEG:
    """Re-encode with a random JPEG quality to simulate compression artifacts."""
    def __init__(self, p=0.25, quality=(30, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomElasticWarp:
    """Smooth elastic distortion (wobbly handwriting) via cv2.remap.

    Random displacements on a coarse grid are upscaled with bicubic interpolation, giving a
    smooth field in ~1 ms. (torchvision's ElasticTransform blurs a full-res field with a huge
    gaussian kernel: ~75 ms/img in a single-threaded DataLoader worker.)
    alpha = max displacement in pixels, grid = coarse grid size (smaller -> smoother bends).
    """
    def __init__(self, p=0.3, alpha=(3.0, 8.0), grid=(4, 8)):
        self.p = p
        self.alpha = alpha
        self.grid = grid

    def __call__(self, img):
        if random.random() > self.p:
            return img
        a = np.array(img)
        h, w = a.shape[:2]
        g = random.randint(*self.grid)
        alpha = random.uniform(*self.alpha)
        # seed from `random` (reseeded per DataLoader worker) so workers don't produce identical warps
        rng = np.random.default_rng(random.getrandbits(32))
        dx = cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        out = cv2.remap(a, xs + dx * alpha, ys + dy * alpha, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)
        return Image.fromarray(out)


class RandomGaussianBlur:
    """Gaussian blur with random radius (out-of-focus / ink bleed)."""
    def __init__(self, p=0.3, radius=(0.5, 1.8)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius)))


# --------------------------------------------------------------------------------------------
# Tensor-stage transforms (applied after ToTensor, values in [0, 1], before Normalize)
# --------------------------------------------------------------------------------------------
class RandomGaussianNoise:
    """Add pixel-wise gaussian noise (sensor / scan noise)."""
    def __init__(self, p=0.3, std=(0.02, 0.08)):
        self.p = p
        self.std = std

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * random.uniform(*self.std)
        return (tensor + noise).clamp(0.0, 1.0)


class RandomSaltPepper:
    """Flip a small fraction of pixels to black or white (dust / speckles)."""
    def __init__(self, p=0.15, amount=(0.005, 0.03)):
        self.p = p
        self.amount = amount

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        amount = random.uniform(*self.amount)
        mask = torch.rand(tensor.shape[1:])
        tensor = tensor.clone()
        tensor[:, mask < amount / 2] = 0.0          # pepper
        tensor[:, mask > 1 - amount / 2] = 1.0      # salt
        return tensor


# --------------------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------------------
train_transform = transforms.Compose([
    # --- native resolution ---
    RandomStrokeThickness(p=0.4),
    RandomLowResolution(p=0.15),
    RandomJPEG(p=0.2),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # --- geometric: rotate / shift / scale / shear ---
    # glyphs fill the frame edge-to-edge, so only zoom OUT (scale <= 1) and shift a little;
    # zooming in / big shifts crop off tails & heads that distinguish e.g. ป vs บ
    transforms.RandomAffine(
        degrees=15,
        translate=(0.05, 0.05),
        scale=(0.8, 1.0),
        shear=(-15, 15, -5, 5),     # mostly horizontal shear (slanted handwriting)
        fill=WHITE,
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=WHITE),
    RandomElasticWarp(p=0.3),

    # --- photometric ---
    RandomGaussianBlur(p=0.25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),

    transforms.ToTensor(),

    # --- tensor noise / occlusion ---
    RandomGaussianNoise(p=0.3),
    RandomSaltPepper(p=0.15),
    # cutout with white = missing / broken stroke parts (kept small: big holes erase the
    # small details that separate similar classes)
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=1.0),
    # cutout with black = ink blot / occluding mark (rarer)
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.04), ratio=(0.5, 2.0), value=0.0),

    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
