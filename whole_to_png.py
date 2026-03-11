import numpy as np
from PIL import Image
import argparse
from math import ceil, sqrt


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def ensure_slices_first(vol: np.ndarray) -> np.ndarray:
    """
    Accept (S,H,W) or (H,W,S). Return (S,H,W).
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    # If first dim looks like height and last dim looks like slices, move slices to front
    # Heuristic: if vol.shape[0] and vol.shape[1] are "image-like" (e.g., 128-1024)
    # and vol.shape[2] is smaller-ish (e.g., <= 512), treat last dim as slices.
    if vol.shape[2] != vol.shape[1] and vol.shape[2] != vol.shape[0]:
        # ambiguous; keep as-is
        return vol

    # Better heuristic for MRNet: often (S, 256, 256). If last dim is 256 and first dim is also 256,
    # could be (256,256,S). If first two dims are equal and third differs, treat third as slices.
    if vol.shape[0] == vol.shape[1] and vol.shape[2] != vol.shape[0]:
        return np.transpose(vol, (2, 0, 1))

    # If shape is (H,W,S) with H=W and S != H, the above catches it.
    # Otherwise assume it's already (S,H,W)
    return vol


def make_montage(vol_slices_first: np.ndarray, cols: int | None = None) -> np.ndarray:
    """
    vol_slices_first: (S,H,W) -> montage (rows*H, cols*W)
    """
    S, H, W = vol_slices_first.shape

    # Default: near-square grid
    if cols is None:
        cols = int(ceil(sqrt(S)))
    cols = max(1, cols)
    rows = int(ceil(S / cols))

    canvas = np.zeros((rows * H, cols * W), dtype=np.float32)
    for i in range(S):
        r = i // cols
        c = i % cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = vol_slices_first[i]

    return canvas


def main():
    ap = argparse.ArgumentParser(description="Tile all slices in a 3D .npy volume into a single montage PNG.")
    ap.add_argument("input", help="Input .npy path (3D volume)")
    ap.add_argument("output", help="Output .png path")
    ap.add_argument("--cols", type=int, default=None, help="Number of columns in montage (default: auto)")
    args = ap.parse_args()

    vol = np.load(args.input)
    print("Loaded:", args.input, "shape:", vol.shape, "dtype:", vol.dtype)

    vol = ensure_slices_first(vol)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume after reshaping, got {vol.shape}")

    montage = make_montage(vol, cols=args.cols)
    montage_u8 = normalize_to_uint8(montage)

    Image.fromarray(montage_u8, mode="L").save(args.output)
    print("Saved montage PNG:", args.output)


if __name__ == "__main__":
    main()