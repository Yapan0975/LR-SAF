"""
LR-SAF: Frozen DeepLabV3-ResNet50 semantic feature extractor.

Used to enrich the confidence head with per-segment semantic context.
Output is the 21-class softmax over PASCAL-VOC categories (background +
20 common classes). Even though VOC classes don't perfectly match
YorkUrban scenes, the dense logits provide useful disambiguation
(e.g., person/car/bottle vs. man-made structure regions).

For final TIP submission we'd switch to a model trained on Cityscapes or
ADE20K with more relevant classes (wall, building, sky, etc.), but for
the first experiment this validates whether semantic features help at all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
)


class SemanticExtractor(nn.Module):
    """Frozen DeepLabV3 wrapper. Returns 21-channel softmax map per image."""

    NUM_CLASSES = 21  # PASCAL VOC + background
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, freeze=True, return_softmax=True):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights)
        self.return_softmax = return_softmax
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, H, W] ImageNet-normalized RGB.
           Returns [B, 21, H, W] softmax map.
        """
        out = self.model(x)['out']    # [B, 21, H, W] logits
        if self.return_softmax:
            return F.softmax(out, dim=1)
        return out


def sample_along_line(feat_map, lines, n_samples=16):
    """Mean-pool feat_map along each line segment.

    Args:
        feat_map: [C, H, W] tensor
        lines:    [N, 4]   (x1, y1, x2, y2) in image coords matching feat_map
        n_samples: number of equidistant samples along each segment

    Returns:
        pooled : [N, C] mean feature per segment
    """
    if isinstance(lines, np.ndarray):
        lines = torch.from_numpy(lines).float()
    lines = lines.to(feat_map.device).float()
    C, H, W = feat_map.shape
    N = lines.shape[0]
    if N == 0:
        return torch.zeros(0, C, device=feat_map.device)

    p1 = lines[:, :2]
    p2 = lines[:, 2:4]
    t = torch.linspace(0, 1, n_samples, device=feat_map.device).unsqueeze(0)
    sx = (p1[:, 0:1] + t * (p2[:, 0:1] - p1[:, 0:1])).clamp(0, W - 1)
    sy = (p1[:, 1:2] + t * (p2[:, 1:2] - p1[:, 1:2])).clamp(0, H - 1)
    sx_i = sx.long(); sy_i = sy.long()
    # Index [C, H, W] -> [N, S, C]
    sampled = feat_map[:, sy_i, sx_i]                     # [C, N, S]
    sampled = sampled.permute(1, 2, 0)                    # [N, S, C]
    return sampled.mean(dim=1)                            # [N, C]


import numpy as np  # late import (only needed in sample_along_line for type guard)

if __name__ == '__main__':
    print("=== Semantic Extractor smoke test ===")
    ext = SemanticExtractor().cuda().eval()
    img = torch.randn(2, 3, 320, 320).cuda()
    out = ext(img)
    print(f"output: {tuple(out.shape)}, sum-per-pixel close to 1: "
          f"mean={out.sum(dim=1).mean().item():.4f}")

    # Sample along lines
    feat = out[0]
    lines = torch.tensor([[10., 10., 100., 100.],
                          [50., 200., 250., 200.],
                          [200., 50., 200., 250.]])
    pooled = sample_along_line(feat, lines, n_samples=16)
    print(f"pooled features: {tuple(pooled.shape)}")
    print(f"first segment top-3 classes: ", end='')
    top3 = pooled[0].topk(3)
    for v, i in zip(top3.values, top3.indices):
        print(f"{ext.VOC_CLASSES[i]}={v.item():.3f}", end=' ')
    print()
