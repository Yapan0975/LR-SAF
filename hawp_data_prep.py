"""Convert YorkUrbanDB into HAWP's expected data/york/ format.

Output:
  data/york/images/<name>.png      (or jpg)
  data/york/test.json              (list of dicts: filename, height, width, lines, junc)
"""
import os, sys, json, glob
import numpy as np
import cv2
import scipy.io as sio

YORK_SRC = '/home/server/Documents/yping/LR-SAF-LSD/data/YorkUrbanDB'
HAWP_ROOT = '/home/server/Documents/yping/LR-SAF-LSD/code/hawp_baseline'
DST_IMG  = os.path.join(HAWP_ROOT, 'data/york/images')
DST_JSON = os.path.join(HAWP_ROOT, 'data/york/test.json')


def read_lines(mat_path):
    """YorkUrban .mat 'lines' is (2N, 2) — every pair of rows is one segment."""
    d = sio.loadmat(mat_path)
    arr = np.asarray(d['lines'], dtype=np.float32)
    if arr.shape[1] == 2 and arr.shape[0] % 2 == 0:
        return np.hstack([arr[0::2], arr[1::2]])
    return arr


def main():
    os.makedirs(DST_IMG, exist_ok=True)
    items = []
    dirs = sorted([d for d in glob.glob(os.path.join(YORK_SRC, '*'))
                    if os.path.isdir(d)])
    print(f"found {len(dirs)} YorkUrban dirs")

    for d in dirs:
        name = os.path.basename(d)
        jpg_src = os.path.join(d, f"{name}.jpg")
        mat = os.path.join(d, f"{name}LinesAndVP.mat")
        if not (os.path.exists(jpg_src) and os.path.exists(mat)):
            continue

        # Copy image (HAWP uses skimage.io which handles jpg and png fine)
        dst_name = f"{name}.jpg"
        dst_path = os.path.join(DST_IMG, dst_name)
        if not os.path.exists(dst_path):
            # symlink to save space
            try:
                os.symlink(jpg_src, dst_path)
            except FileExistsError:
                pass

        # Read image to get dimensions
        img = cv2.imread(jpg_src)
        H, W = img.shape[:2]

        # Read lines + extract junctions from unique endpoints
        lines = read_lines(mat)                  # [N, 4]
        endpoints = np.vstack([lines[:, :2], lines[:, 2:]])   # [2N, 2]
        # Cluster endpoints within 3 px to get junctions
        kept = []
        used = np.zeros(len(endpoints), dtype=bool)
        for i in range(len(endpoints)):
            if used[i]: continue
            d2 = ((endpoints - endpoints[i]) ** 2).sum(-1)
            close = (d2 < 9.0)                   # threshold = 3 px
            kept.append(endpoints[close].mean(axis=0))
            used[close] = True
        junc = np.asarray(kept, dtype=np.float32)

        items.append({
            'filename': dst_name,
            'height':   int(H),
            'width':    int(W),
            'lines':    lines.tolist(),
            'junc':     junc.tolist(),
        })

    with open(DST_JSON, 'w') as f:
        json.dump(items, f)
    print(f"saved {DST_JSON} with {len(items)} entries")
    print(f"images at {DST_IMG}")

    # Quick stats
    print(f"\nstats:")
    print(f"  median #lines / image: {int(np.median([len(i['lines']) for i in items]))}")
    print(f"  median #juncs / image: {int(np.median([len(i['junc']) for i in items]))}")
    print(f"  median image size:     "
          f"{int(np.median([i['width'] for i in items]))} x "
          f"{int(np.median([i['height'] for i in items]))}")


if __name__ == '__main__':
    main()
