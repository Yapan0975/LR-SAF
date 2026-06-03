"""
DINOv2 ViT-S/14 self-supervised features as a semantic prior.

Compared to VOC-21 DeepLabV3:
  - 384-d general features (vs 21-d class probs)
  - Self-supervised on 142M images (LVD-142M), no class bias
  - Robust across natural / man-made / urban scenes
  - Same forward speed as DeepLabV3-ResNet50 (~85MB)

Output is dense patch features at stride 14, upsampled to image resolution
via bilinear interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT-S/14. Returns 384-channel dense feature map."""

    def __init__(self, weights_path=None, freeze=True, out_channels=384):
        super().__init__()
        # timm has the DINOv2 architecture
        import timm
        # Build with pretrained=False, then load custom weights
        # Use dynamic_img_size so we can pass arbitrary multiples of 14
        self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m',
                                        pretrained=False,
                                        num_classes=0,
                                        img_size=518,
                                        dynamic_img_size=True)
        if weights_path is None:
            weights_path = os.path.expanduser(
                '~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu', weights_only=False)
            # The fb-released DINOv2 state_dict keys are slightly different from timm
            # Use a remap: drop missing/unexpected unless strict
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if len(missing) > 5 or len(unexpected) > 5:
                # Try to adapt key names from fb -> timm
                state = self._adapt_fb_to_timm(state)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
            self._missing = missing
            self._unexpected = unexpected
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
        self.model.eval()
        self.out_channels = out_channels
        self.patch_size = 14

    @staticmethod
    def _adapt_fb_to_timm(state):
        """Minimal key remapping fb DINOv2 -> timm ViT.

        Most keys actually match between fb DINOv2 release and timm. Only
        a handful differ; we patch those.
        """
        out = {}
        for k, v in state.items():
            nk = k
            # fb uses "patch_embed.proj"; timm same
            # fb uses "blocks.{i}.attn.qkv"; timm same
            # fb uses "norm" for final layernorm; timm uses "fc_norm" or "norm"
            # Keep most as-is; remap a few common variants
            if k.startswith('module.'):
                nk = k[len('module.'):]
            out[nk] = v
        return out

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, H, W] ImageNet-normalized.
        Returns: [B, 384, H, W] dense feature map (upsampled from patch grid).
        """
        B, C, H, W = x.shape
        # Resize to a multiple of patch_size
        h_p = (H // self.patch_size) * self.patch_size
        w_p = (W // self.patch_size) * self.patch_size
        if (h_p != H) or (w_p != W):
            x_r = F.interpolate(x, size=(h_p, w_p), mode='bilinear',
                                align_corners=False)
        else:
            x_r = x
        # forward features
        feats = self.model.forward_features(x_r)
        if isinstance(feats, dict):
            patch_tokens = feats.get('x', feats.get('patch_tokens'))
        else:
            # Tensor of shape [B, 1+N, D] with CLS token at index 0 if present
            n_total = feats.shape[1]
            grid_h = h_p // self.patch_size
            grid_w = w_p // self.patch_size
            expected = grid_h * grid_w
            if n_total == expected + 1:
                patch_tokens = feats[:, 1:]
            elif n_total == expected:
                patch_tokens = feats
            else:
                # Probably has reg tokens; strip CLS + N register tokens at start
                extra = n_total - expected
                patch_tokens = feats[:, extra:]
        Dch = patch_tokens.shape[-1]
        grid_h = h_p // self.patch_size
        grid_w = w_p // self.patch_size
        feat_map = patch_tokens.transpose(1, 2).reshape(B, Dch, grid_h, grid_w)
        feat_map = F.interpolate(feat_map, size=(H, W), mode='bilinear',
                                  align_corners=False)
        return feat_map


if __name__ == '__main__':
    print("=== DINOv2 Extractor smoke test ===")
    ext = DINOv2Extractor().cuda()
    print(f"loaded; missing={len(ext._missing) if hasattr(ext, '_missing') else '?'} "
          f"unexpected={len(ext._unexpected) if hasattr(ext, '_unexpected') else '?'}")
    if hasattr(ext, '_missing') and ext._missing[:3]:
        print(f"  missing samples: {ext._missing[:3]}")
    if hasattr(ext, '_unexpected') and ext._unexpected[:3]:
        print(f"  unexpected samples: {ext._unexpected[:3]}")
    x = torch.randn(2, 3, 320, 320).cuda()
    out = ext(x)
    print(f"output: {tuple(out.shape)}")
    print(f"feature stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
