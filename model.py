"""
LR-SAF: Network wrapper around the AFM a-trous Residual U-Net.

We reuse the AFM backbone but expand the output head from 2 -> 4 channels:
  - 2 channels: (a_x, a_y) attraction vector (bounded)
  - 1 channel : t_star projection parameter (sigmoid -> [0, 1])
  - 1 channel : junction strength (sigmoid -> [0, 1])

This lets us load AFM's pretrained backbone weights for hot-start and only
train the new heads + TNNR regularizer.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_afm_on_path():
    afm_root = '/home/server/Documents/yping/LR-SAF-LSD/code/afm_baseline'
    if afm_root not in sys.path:
        sys.path.insert(0, afm_root)
        sys.path.insert(0, afm_root + '/lib')


class LRSAFNet(nn.Module):
    """Wrap AFM's DeepLabv3+ a-trous backbone with a 4-channel output head."""

    def __init__(self, afm_backbone, hidden_ch=64):
        super().__init__()
        self.backbone = afm_backbone

        # Locate the final conv (output_channels = 2 in vanilla AFM)
        # and replace it with our 4-channel head.
        last_layer_name, last_layer = self._find_last_conv(afm_backbone)
        assert isinstance(last_layer, nn.Conv2d), \
            f"expected last layer to be Conv2d, got {type(last_layer)}"
        in_ch = last_layer.in_channels
        self._last_layer_name = last_layer_name

        # Replace the final 1x1 conv with our 4-channel head
        new_head = nn.Conv2d(in_ch, 4, kernel_size=1, stride=1, bias=True)
        # Initialize: copy first 2 channels from the pretrained final layer
        with torch.no_grad():
            new_head.weight[:2].copy_(last_layer.weight)
            new_head.weight[2:].normal_(mean=0.0, std=1e-3)
            if last_layer.bias is not None:
                new_head.bias[:2].copy_(last_layer.bias)
            new_head.bias[2:].zero_()

        # Plug it in
        self._replace_module(afm_backbone, last_layer_name, new_head)
        self.head = new_head

    @staticmethod
    def _find_last_conv(module):
        """Recurse and find the last Conv2d layer with output channels == 2."""
        last = None
        last_name = None
        for name, child in module.named_modules():
            if isinstance(child, nn.Conv2d) and child.out_channels == 2:
                last, last_name = child, name
        if last is None:
            # Fall back to last Conv2d of any shape
            for name, child in module.named_modules():
                if isinstance(child, nn.Conv2d):
                    last, last_name = child, name
        return last_name, last

    @staticmethod
    def _replace_module(root, dotted_name, new_module):
        parts = dotted_name.split('.')
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    def forward(self, x):
        """Returns a dict with keys 'a' [B,2,H,W], 't_star' [B,1,H,W], 'junc' [B,1,H,W]."""
        out = self.backbone(x)                  # [B, 4, H, W]
        if isinstance(out, (list, tuple)):
            out = out[0]
        # a_x, a_y are raw outputs (not yet bounded; bounded encoding done outside)
        a = out[:, :2]                          # [B, 2, H, W]
        t_star = torch.sigmoid(out[:, 2:3])     # [B, 1, H, W]
        junc = torch.sigmoid(out[:, 3:4])       # [B, 1, H, W]
        return {'a': a, 't_star': t_star, 'junc': junc}


def build_lr_saf(load_afm_pretrained=True,
                 afm_ckpt='/home/server/Documents/yping/LR-SAF-LSD/checkpoints/atrous/weight/model_final.pth.tar',
                 afm_cfg='/home/server/Documents/yping/LR-SAF-LSD/code/afm_baseline/experiments/afm_atrous.yaml',
                 device='cuda'):
    """Build LR-SAF model with optional AFM pretrained backbone."""
    _ensure_afm_on_path()
    import os as _os
    _os.chdir('/home/server/Documents/yping/LR-SAF-LSD/code/afm_baseline')
    from config import cfg
    cfg.merge_from_file(afm_cfg)
    from modeling.net import build_network
    backbone = build_network(cfg)

    if load_afm_pretrained:
        ckpt = torch.load(afm_ckpt, map_location='cpu', weights_only=False)
        backbone.load_state_dict(ckpt, strict=True)
        print(f"Loaded AFM pretrained from {afm_ckpt}")

    model = LRSAFNet(backbone).to(device)
    return model


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, '/home/server/Documents/yping/LR-SAF-LSD/code/lr_saf')
    model = build_lr_saf(device='cuda').eval()
    x = torch.randn(2, 3, 320, 320, device='cuda')
    with torch.no_grad():
        out = model(x)
    print("model output shapes:")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}, range=[{v.min().item():.3f}, {v.max().item():.3f}]")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params / 1e6:.2f} M")
