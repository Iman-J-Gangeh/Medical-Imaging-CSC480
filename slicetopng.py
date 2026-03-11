import numpy as np
from PIL import Image
import argparse
from pathlib import Path


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def save_slice(slice2d: np.ndarray, out_path: Path):
    img = normalize_to_uint8(slice2d)
    Image.fromarray(img, mode="L").save(out_path)


def make_montage(volume: np.ndarray, cols: int = 8) -> np.ndarray:
    """
    volume: (S, H, W) -> returns a big 2D montage image
    """
    S, H, W = volume.shape
    cols = max(1, cols)
    rows = int(np.ceil(S / cols))

    canvas = np.zeros((rows * H, cols * W), dtype=volume.dtype)
    for i in range(S):
        r = i // cols
        c = i % cols
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = volume[i]
    return canvas


def convert_npy_to_png(input_path: str, output_path: str, mode: str, slice_index: int | None, montage_cols: int):
    arr = np.load(input_path)
    print("Loaded shape:", arr.shape, "dtype:", arr.dtype)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Case 1: single grayscale image
    if arr.ndim == 2:
        save_slice(arr, out_path)
        print("Saved:", out_path)
        return

    # Case 2: volume (S, H, W) — typical MRNet
    if arr.ndim == 3 and arr.shape[0] != 3 and arr.shape[0] != 1:
        vol = arr  # (S, H, W)

        if mode == "middle":
            idx = vol.shape[0] // 2 if slice_index is None else slice_index
            idx = max(0, min(idx, vol.shape[0] - 1))
            save_slice(vol[idx], out_path)
            print(f"Saved middle slice (index {idx}) to:", out_path)
            return

        if mode == "all":
            stem = out_path.stem
            suffix = out_path.suffix or ".png"
            folder = out_path.parent
            for i in range(vol.shape[0]):
                p = folder / f"{stem}_slice{i:03d}{suffix}"
                save_slice(vol[i], p)
            print(f"Saved {vol.shape[0]} slices to folder:", folder)
            return

        if mode == "montage":
            montage = make_montage(vol, cols=montage_cols)
            save_slice(montage, out_path)
            print(f"Saved montage ({vol.shape[0]} slices) to:", out_path)
            return

        raise ValueError("Unknown mode. Use middle|all|montage")

    # Case 3: image with channels last (H, W, C)
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        if arr.shape[2] == 1:
            save_slice(arr[:, :, 0], out_path)
        else:
            img = normalize_to_uint8(arr)
            Image.fromarray(img, mode="RGB").save(out_path)
        print("Saved:", out_path)
        return

    raise ValueError(f"Unsupported array shape: {arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy to .png (supports MRI volumes)")
    parser.add_argument("input", help="Path to input .npy file")
    parser.add_argument("output", help="Path to output .png file")

    parser.add_argument("--mode", choices=["middle", "all", "montage"], default="middle",
                        help="For 3D volumes (S,H,W): save middle slice, all slices, or a montage")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice index to save (only used with --mode middle)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Montage columns (only used with --mode montage)")

    args = parser.parse_args()

    convert_npy_to_png(args.input, args.output, args.mode, args.slice, args.cols)