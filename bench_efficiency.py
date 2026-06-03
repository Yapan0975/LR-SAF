"""Measure FPS / Params / FLOPs for the three models."""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')
os.chdir(ROOT + '/code/afm_baseline')

import torch
import numpy as np
from config import cfg
cfg.merge_from_file('experiments/afm_atrous.yaml')
from modeling.net import build_network
from model import build_lr_saf
from semantic_features import SemanticExtractor


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, input_shape=(1, 3, 320, 320), is_dict_out=False):
    """Approximate FLOPs via hook on conv/linear layers."""
    flops = [0]
    def conv_hook(m, inp, out):
        # FLOPs = output_elements * (kernel * in_ch / groups + bias)
        if hasattr(m, 'kernel_size'):
            k = m.kernel_size[0] * m.kernel_size[1] if len(m.kernel_size) == 2 else m.kernel_size[0]
            o = out.numel()
            flops[0] += o * (k * m.in_channels / max(getattr(m, 'groups', 1), 1))
            if m.bias is not None: flops[0] += o
    def lin_hook(m, inp, out):
        flops[0] += out.numel() * m.in_features
        if m.bias is not None: flops[0] += out.numel()
    handles = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
    x = torch.randn(input_shape).cuda()
    with torch.no_grad():
        _ = model(x)
    for h in handles: h.remove()
    return flops[0]


def measure_fps(model, n_warm=10, n_run=50, input_shape=(1, 3, 320, 320),
                 extra_calls=None):
    x = torch.randn(input_shape).cuda()
    # Warm-up
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(x)
            if extra_calls: [c(x) for c in extra_calls]
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_run):
            _ = model(x)
            if extra_calls: [c(x) for c in extra_calls]
    torch.cuda.synchronize()
    return n_run / (time.time() - t0)


print("=== EFFICIENCY BENCHMARK (RTX 5090, input 320x320, batch=1) ===\n")

# 1. AFM baseline
afm = build_network(cfg).cuda().eval()
ckpt = torch.load(ROOT + '/checkpoints/atrous/weight/model_final.pth.tar',
                  map_location='cuda', weights_only=False)
afm.load_state_dict(ckpt, strict=True)

# 2. LR-SAF
lrsaf = build_lr_saf(device='cuda').eval()

# 3. Semantic extractor
sem = SemanticExtractor().cuda().eval()

# Measure params
p_afm = count_params(afm)
p_lr = count_params(lrsaf)
p_sem = count_params(sem)
print(f"Params:")
print(f"  AFM baseline      : {p_afm/1e6:.2f} M")
print(f"  LR-SAF main       : {p_lr/1e6:.2f} M")
print(f"  Semantic (frozen) : {p_sem/1e6:.2f} M")
print(f"  LR-SAF + sem total: {(p_lr + p_sem)/1e6:.2f} M")

# Measure FLOPs
print(f"\nFLOPs (1x320x320):")
flops_afm = measure_flops(afm)
flops_lr = measure_flops(lrsaf)
flops_sem = measure_flops(sem)
print(f"  AFM baseline       : {flops_afm/1e9:.2f} G")
print(f"  LR-SAF main        : {flops_lr/1e9:.2f} G")
print(f"  Semantic           : {flops_sem/1e9:.2f} G")
print(f"  LR-SAF + sem total : {(flops_lr + flops_sem)/1e9:.2f} G")

# Measure FPS
print(f"\nFPS (forward only, GPU sync):")
fps_afm = measure_fps(afm)
fps_lr = measure_fps(lrsaf)
fps_lr_sem = measure_fps(lrsaf, extra_calls=[sem])
print(f"  AFM baseline       : {fps_afm:.1f} FPS ({1000/fps_afm:.1f} ms)")
print(f"  LR-SAF main        : {fps_lr:.1f} FPS ({1000/fps_lr:.1f} ms)")
print(f"  LR-SAF + semantic  : {fps_lr_sem:.1f} FPS ({1000/fps_lr_sem:.1f} ms)")

# Table 6 ready
print("\n=== MARKDOWN TABLE 6 ===\n")
print("| Model | Params (M) | FLOPs (G) | FPS (RTX 5090) | Latency (ms) |")
print("|---|---|---|---|---|")
print(f"| AFM baseline | {p_afm/1e6:.2f} | {flops_afm/1e9:.2f} | {fps_afm:.1f} | {1000/fps_afm:.1f} |")
print(f"| LR-SAF (main) | {p_lr/1e6:.2f} | {flops_lr/1e9:.2f} | {fps_lr:.1f} | {1000/fps_lr:.1f} |")
print(f"| LR-SAF + sem head | {(p_lr + p_sem)/1e6:.2f} | {(flops_lr + flops_sem)/1e9:.2f} | {fps_lr_sem:.1f} | {1000/fps_lr_sem:.1f} |")
