"""

Compute how diverse a set of **human-validated fault-revealing images** are from each other.

Diversity is the mean pairwise LPIPS across all images selected.


  # All PNGs in a folder
  python folder_diversity.py --dir /path/to/faults --ext png

"""

import argparse
import json
import itertools
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import torch

try:
    import lpips
except Exception as e:
    raise SystemExit("ERROR: lpips is not installed. Try: pip install lpips")


def list_images(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    exts_norm = {e.lower().lstrip(".") for e in exts}
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts_norm]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in exts_norm]
    files.sort()
    return files


def load_rgb_uint8(p: Path) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    return np.array(im, dtype=np.uint8)


def to_lpips_tensor(arr_uint8_hwc: np.ndarray, min_side: int = 64) -> torch.Tensor:
    t = torch.from_numpy(arr_uint8_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # 1x3xHxW
    H, W = t.shape[-2:]
    if H < min_side or W < min_side:
        t = torch.nn.functional.interpolate(t, size=(max(H, min_side), max(W, min_side)),
                                            mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0  # [-1,1]


def compute_pairwise_lpips(tensors: List[torch.Tensor],
                           net: "lpips.LPIPS",
                           device: torch.device,
                           max_pairs: Optional[int] = None,
                           seed: int = 42):
    n = len(tensors)
    if n < 2:
        return {
            "n_images": n, "pairs": 0,
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
        }
    idx_pairs = list(itertools.combinations(range(n), 2))

    if (max_pairs is not None) and (len(idx_pairs) > max_pairs):
        import random
        random.Random(seed).shuffle(idx_pairs)
        idx_pairs = idx_pairs[:max_pairs]

    vals = []
    with torch.no_grad():
        for i, j in idx_pairs:
            d = net(tensors[i].to(device), tensors[j].to(device))
            vals.append(float(d.item()))

    arr = np.array(vals, dtype=np.float64)
    return {
        "n_images": n,
        "pairs": int(len(idx_pairs)),
        "mean": float(arr.mean() if arr.size else 0.0),
        "std": float(arr.std(ddof=0) if arr.size > 1 else 0.0),
        "min": float(arr.min() if arr.size else 0.0),
        "max": float(arr.max() if arr.size else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="Folder containing fault-revealing images.")
    ap.add_argument("--ext", type=str, nargs="+", default=["png"], help="Image extensions to include (e.g., png jpg).")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under --dir.")
    ap.add_argument("--min_side", type=int, default=64, help="Resize shortest side to at least this before LPIPS.")
    ap.add_argument("--lpips_net", type=str, choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone.")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto-select if omitted).")
    ap.add_argument("--max_pairs", type=int, default=None, help="Subsample at most this many pairs for speed.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for pair subsampling.")
    ap.add_argument("--save_csv", type=str, default=None, help="Optional CSV output path.")
    ap.add_argument("--save_json", type=str, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ERROR: directory not found: {root}")


    dev = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(dev)

    print(f"[info] scanning {root} (recursive={args.recursive}) for extensions {args.ext}")
    files = list_images(root, args.ext, args.recursive)
    if not files:
        raise SystemExit("No images found with the given extensions.")

    print(f"[info] found {len(files)} image(s). loading + preprocessing for LPIPS...")
    tensors = [to_lpips_tensor(load_rgb_uint8(p), args.min_side) for p in files]

    print(f"[info] computing pairwise LPIPS on device={device}, net={args.lpips_net} ...")
    net = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    stats = compute_pairwise_lpips(tensors, net, device, max_pairs=args.max_pairs, seed=args.seed)

    print("\n=== Diversity (mean pairwise LPIPS) ===")
    print(f"Images:         {stats['n_images']}")
    print(f"Pairs:          {stats['pairs']}")
    print(f"Mean:           {stats['mean']:.6f}")
    print(f"Std:            {stats['std']:.6f}")
    print(f"Min/Max:        {stats['min']:.6f} / {stats['max']:.6f}")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dir", "n_images", "pairs", "mean", "std", "min", "max",
                        "recursive", "min_side", "lpips_net", "device", "max_pairs"])
            w.writerow([str(root), stats["n_images"], stats["pairs"], stats["mean"], stats["std"],
                        stats["min"], stats["max"], int(args.recursive), args.min_side,
                        args.lpips_net, device.type, args.max_pairs or 0])
        print(f"[ok] wrote CSV: {args.save_csv}")

    if args.save_json:
        out = {
            "dir": str(root),
            "n_images": stats["n_images"],
            "pairs": stats["pairs"],
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "recursive": bool(args.recursive),
            "min_side": int(args.min_side),
            "lpips_net": args.lpips_net,
            "device": device.type,
            "max_pairs": int(args.max_pairs) if args.max_pairs is not None else None,
        }
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[ok] wrote JSON: {args.save_json}")


if __name__ == "__main__":
    main()
