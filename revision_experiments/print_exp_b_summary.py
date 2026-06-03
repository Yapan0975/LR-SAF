"""Print a clean one-line summary of the EXP-B zero-shot JSON."""
import json, sys

with open(sys.argv[1]) as fh:
    d = json.load(fh)
print(f"label   : {d['label']}")
print(f"images  : {d['n_images']}")
print(f"F-mean  : {d['F_mean']:.4f} ± {d['F_std']:.4f}")
print(f"sAP-10  : {d['sAP_10_mean']:.4f} ± {d['sAP_10_std']:.4f}")
print(f"sAP-5   : {d['sAP_5_mean']:.4f}")
print(f"sAP-15  : {d['sAP_15_mean']:.4f}")
