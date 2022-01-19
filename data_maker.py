import random
import cv2
import numpy as np
import rawpy
import argparse
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map

import matplotlib.pyplot as plt
import rawpy



def compute_wb(raw):
    # print("Computing WB for %s" % (raw_path))
    if isinstance(raw, str) or isinstance(raw, Path):
        bayer = rawpy.imread(str(raw))
    else:
        bayer = raw  # rawpy object일 경우

    # bayer = rawpy.imread(raw_path)
    rgb_nowb = bayer.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=16)

    rgb_wb = bayer.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16)

    wb = [
        np.mean(rgb_wb[..., 0]) / np.mean(rgb_nowb[..., 0]),
        np.mean(rgb_wb[..., 1]) / np.mean(rgb_nowb[..., 1]),
        np.mean(rgb_wb[..., 1]) / np.mean(rgb_nowb[..., 1]),
        np.mean(rgb_wb[..., 2]) / np.mean(rgb_nowb[..., 2]),
    ]
    wb = np.array(wb, dtype=np.float32)
    return wb


def get_4ch(bayer):
    h, w = bayer.shape[:2]
    rgba = np.zeros((h // 2, w // 2, 4), dtype=np.float32)
    rgba[:, :, 0] = bayer[0::2, 0::2]  # R
    rgba[:, :, 1] = bayer[1::2, 0::2]  # G1
    rgba[:, :, 2] = bayer[1::2, 1::2]  # B
    rgba[:, :, 3] = bayer[0::2, 1::2]  # G2

    return rgba

# def make_data(raw_path, rgb_path, scale, black_lv=512, white_lv=16383):
def make_data(raw_path, scale, black_lv=512, white_lv=16383):
    with rawpy.imread(str(raw_path)) as r:
        hr_bayer = r.raw_image_visible.astype(np.float32)
        hr_bayer = (hr_bayer - black_lv) / (white_lv - black_lv)
        wb = compute_wb(r)

        hr_rgb = r.postprocess(no_auto_bright=False, use_camera_wb=True, output_bps=8)
        #hr_rgb = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=8)
        # hr_rgb = imageio.imread(rgb_path)
        # hr_bayer = hr_bayer[8:-8, 8:-8]

    hr_raw = get_4ch(hr_bayer)
    h, w = hr_raw.shape[:2]
    lr_raw = cv2.resize(hr_raw, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
    # lr_raw = hr_raw[0::scale, 0::scale]
    # lr_bayer = get_1ch(lr_raw)

    return lr_raw, hr_rgb, wb

def _T_make_data(raw_path, scale, out_dir, num_patches, patch_size):

    lr_raw, hr_rgb, wb = make_data(raw_path, scale)
    #print("Start patching... :", lr_raw.shape)
    for i in range(num_patches):
        h, w = lr_raw.shape[:2]
        p = patch_size
        pos_x, pos_y = random.randint(0, w - p), random.randint(0, h - p)

        lr_raw_patch = lr_raw[pos_y : pos_y + p, pos_x : pos_x + p]
        hr_rgb_patch = hr_rgb[pos_y * scale * 2 : (pos_y + p) * scale * 2, pos_x * scale * 2 : (pos_x + p) * scale * 2]
        hr_rgb_patch = hr_rgb_patch.astype(np.float32) / 255
        fpath = out_dir / f"{raw_path.parent.name}_{raw_path.stem}_{i:02d}.npz"
        np.savez_compressed(fpath, lr_raw=lr_raw_patch, hr_rgb=hr_rgb_patch, wb=wb)

    return raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data maker')

    parser.add_argument('--scale', '-s', type=int, help='scale')
    parser.add_argument('--n_patch', '-p', type=int, help='number of patches per image')
    parser.add_argument('--path', type=str, help="Dir for output save")
    args = parser.parse_args()

    data_dir = Path(r"/data/paul/")
    out_dir = Path(r"/data/paul/")

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / args.path / "test").mkdir(parents=True, exist_ok=True)

    train_files = sorted(list((data_dir / "train" ).rglob("*.ARW")))
    test_files = sorted(list((data_dir / "test" ).rglob("*.ARW")))

    # JPG 파일들. ARW만 있고, JPG는 없는 경우 제외
    train_files = list(filter(lambda file: (file.parent / f"{file.stem}.JPG").exists(), train_files))
    test_files = list(filter(lambda file: (file.parent / f"{file.stem}.JPG").exists(), test_files))

    train_rgb_files = [(file.parent / f"{file.stem}.JPG") for file in train_files]
    test_rgb_files = [(file.parent / f"{file.stem}.JPG") for file in test_files]


    func = partial(_T_make_data, scale=args.scale, out_dir=out_dir / "train", num_patches=args.n_patch, patch_size=96)
    process_map(func, train_files, max_workers=12, chunksize=24, desc="Train data creating...")

    func = partial(_T_make_data, scale=args.scale, out_dir=out_dir / "test", num_patches=args.n_patch, patch_size=96)
    process_map(func, train_files, max_workers=12, chunksize=24, desc="Test data creating...")