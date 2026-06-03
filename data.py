"""
Minimal dataset loader for sanity check.

Uses YorkUrban images + their .mat GT lines as a stand-in until we have
the Wireframe pointlines.zip. YorkUrban GT format:

  P1020171/
    P1020171.jpg
    P1020171LinesAndVP.mat  -> contains 'lines' field [N, 4] (x1, y1, x2, y2)
"""
import os
import os.path as osp
import glob
import numpy as np
import cv2
import torch
import torch.utils.data as data
import scipy.io as sio

YORK_ROOT = '/home/server/Documents/yping/LR-SAF-LSD/data/YorkUrbanDB'


class YorkUrbanSubset(data.Dataset):
    """Tiny YorkUrban loader for LR-SAF sanity check."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, root=YORK_ROOT, in_res=320, limit=None):
        self.root = root
        self.in_res = in_res
        all_dirs = sorted([d for d in glob.glob(osp.join(root, '*'))
                           if osp.isdir(d) and not d.endswith('__MACOSX')])
        if limit:
            all_dirs = all_dirs[:limit]
        self.items = []
        for d in all_dirs:
            name = osp.basename(d)
            jpg = osp.join(d, f"{name}.jpg")
            mat = osp.join(d, f"{name}LinesAndVP.mat")
            if osp.exists(jpg) and osp.exists(mat):
                self.items.append((name, jpg, mat))

    def __len__(self):
        return len(self.items)

    def _read_lines(self, mat_path):
        """YorkUrban stores lines as a (2N, 2) array: each pair of rows is one segment."""
        try:
            data = sio.loadmat(mat_path)
            if 'lines' in data:
                arr = np.asarray(data['lines'], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] % 2 == 0:
                    # (2N, 2) -> (N, 4)
                    lines = np.hstack([arr[0::2], arr[1::2]])
                elif arr.ndim == 2 and arr.shape[1] == 4:
                    lines = arr
                else:
                    raise RuntimeError(f"unexpected shape {arr.shape} in {mat_path}")
                return lines
            raise RuntimeError(f"no 'lines' key in {mat_path}")
        except Exception as e:
            print(f"failed to read {mat_path}: {e}")
            return np.zeros((0, 4), dtype=np.float32)

    def __getitem__(self, idx):
        name, jpg, mat = self.items[idx]
        img = cv2.imread(jpg)                             # BGR
        H_o, W_o = img.shape[:2]
        img_r = cv2.resize(img, (self.in_res, self.in_res)).astype(np.float32) / 255.0
        img_r = (img_r - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float()   # [3, H, W]

        lines = self._read_lines(mat)                              # [N, 4] in original coords
        # Scale to resized image
        sx, sy = self.in_res / W_o, self.in_res / H_o
        lines = lines.copy()
        lines[:, 0::2] *= sx
        lines[:, 1::2] *= sy
        lines_t = torch.from_numpy(lines).float()

        return {
            'name':   name,
            'image':  img_t,
            'lines':  lines_t,
            'H_orig': H_o,
            'W_orig': W_o,
        }


def collate_variable_lines(batch):
    """Custom collate: pad lines to same length within a batch."""
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


if __name__ == '__main__':
    ds = YorkUrbanSubset(limit=5)
    print(f"items found: {len(ds)}")
    for i in range(min(3, len(ds))):
        item = ds[i]
        print(f"  {item['name']}: image {tuple(item['image'].shape)}, "
              f"lines {tuple(item['lines'].shape)}, "
              f"line range: x=[{item['lines'][:,0::2].min().item():.1f}, {item['lines'][:,0::2].max().item():.1f}], "
              f"y=[{item['lines'][:,1::2].min().item():.1f}, {item['lines'][:,1::2].max().item():.1f}]")
