"""
Dataset loader for HAWP-format JSON annotations (Wireframe + YorkUrban).

Format (HAWP's `TestDatasetWithAnnotations`):
  annotations[i] = {
    'filename': '...jpg',
    'height': int,
    'width': int,
    'lines': [[x1, y1, x2, y2], ...],
    'junc':  [[x, y], ...]   (optional)
  }

This loader mirrors YorkUrbanSubset's interface so train_full.py / train_ablation.py
work without changes.
"""
import os
import os.path as osp
import json
import numpy as np
import cv2
import torch
import torch.utils.data as data

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HAWPJsonDataset(data.Dataset):
    """Generic HAWP-format dataset (works for Wireframe / YorkUrban)."""

    def __init__(self, ann_file, img_dir, in_res=320, limit=None,
                 augment=False):
        self.img_dir = img_dir
        with open(ann_file) as f:
            self.items = json.load(f)
        if limit is not None:
            self.items = self.items[:limit]
        self.in_res = in_res
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ann = self.items[idx]
        path = osp.join(self.img_dir, ann['filename'])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        H_o, W_o = img.shape[:2]
        # Standardize to in_res x in_res
        img_r = cv2.resize(img, (self.in_res, self.in_res)).astype(np.float32) / 255.0
        img_r = (img_r - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float()

        # Wireframe train uses (junctions, edges_positive); test uses (lines)
        if 'lines' in ann:
            lines = np.asarray(ann['lines'], dtype=np.float32)
        elif 'junctions' in ann and 'edges_positive' in ann:
            juncs = np.asarray(ann['junctions'], dtype=np.float32)
            edges = np.asarray(ann['edges_positive'], dtype=np.int64)
            if len(edges) > 0:
                lines = np.concatenate([juncs[edges[:, 0]], juncs[edges[:, 1]]],
                                        axis=1)   # [N, 4]
            else:
                lines = np.zeros((0, 4), dtype=np.float32)
        else:
            lines = np.zeros((0, 4), dtype=np.float32)
        # Scale to in_res frame
        if len(lines) > 0:
            sx, sy = self.in_res / W_o, self.in_res / H_o
            lines = lines.copy()
            lines[:, 0::2] *= sx
            lines[:, 1::2] *= sy
        else:
            lines = np.zeros((0, 4), dtype=np.float32)

        if self.augment:
            img_t, lines = self._augment(img_t, lines)

        return {
            'name':   osp.splitext(ann['filename'])[0],
            'image':  img_t,
            'lines':  torch.from_numpy(lines).float(),
            'H_orig': H_o,
            'W_orig': W_o,
        }

    def _augment(self, img_t, lines):
        """Random h/v flip + 90/180/270 rotation, matching AFM convention."""
        # Horizontal flip
        if np.random.rand() < 0.5:
            img_t = torch.flip(img_t, dims=[2])    # flip W
            if len(lines) > 0:
                lines = lines.copy()
                lines[:, 0::2] = self.in_res - 1 - lines[:, 0::2]
        # Vertical flip
        if np.random.rand() < 0.5:
            img_t = torch.flip(img_t, dims=[1])    # flip H
            if len(lines) > 0:
                lines = lines.copy()
                lines[:, 1::2] = self.in_res - 1 - lines[:, 1::2]
        return img_t, lines


def collate_variable_lines(batch):
    """Pad lines to same length within a batch."""
    images = torch.stack([b['image'] for b in batch])
    max_n = max((b['lines'].shape[0] for b in batch), default=0)
    if max_n == 0:
        lines = torch.zeros(len(batch), 0, 4)
        mask = torch.zeros(len(batch), 0, dtype=torch.bool)
    else:
        lines = torch.zeros(len(batch), max_n, 4)
        mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b['lines'].shape[0]
            lines[i, :n] = b['lines']
            mask[i, :n] = True
    return {
        'image': images,
        'lines': lines,
        'lines_mask': mask,
        'name':  [b['name'] for b in batch],
    }


# Convenience factories
def wireframe_train(root, in_res=320, augment=True):
    return HAWPJsonDataset(
        ann_file=osp.join(root, 'wireframe/train.json'),
        img_dir =osp.join(root, 'wireframe/images'),
        in_res=in_res, augment=augment,
    )


def wireframe_test(root, in_res=320):
    return HAWPJsonDataset(
        ann_file=osp.join(root, 'wireframe/test.json'),
        img_dir =osp.join(root, 'wireframe/images'),
        in_res=in_res, augment=False,
    )


def york_test(root, in_res=320):
    return HAWPJsonDataset(
        ann_file=osp.join(root, 'york/test.json'),
        img_dir =osp.join(root, 'york/images'),
        in_res=in_res, augment=False,
    )


if __name__ == '__main__':
    import sys
    DATA_ROOT = '/home/server/Documents/yping/LR-SAF-LSD/code/hawp_baseline/data'
    if not osp.isdir(DATA_ROOT):
        print(f"data not yet extracted at {DATA_ROOT}")
        sys.exit(0)
    for kind, ds_factory in [('wireframe_train', wireframe_train),
                              ('wireframe_test', wireframe_test),
                              ('york_test', york_test)]:
        try:
            ds = ds_factory(DATA_ROOT)
            print(f"{kind}: {len(ds)} samples")
            item = ds[0]
            print(f"  first: image {tuple(item['image'].shape)}, "
                  f"lines {tuple(item['lines'].shape)}, "
                  f"size {item['H_orig']}x{item['W_orig']}")
        except Exception as e:
            print(f"{kind}: failed - {e}")
