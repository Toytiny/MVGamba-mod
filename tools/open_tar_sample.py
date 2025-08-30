#!/usr/bin/env python3
import os, tarfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

SRC_DIR = Path("/data/mvgamba_gobj/savedata")     # 所有 .tar 的目录
OUT_DIR = Path("/data/mvgamba_gobj/preview")      # 可视化输出根目录
SAMPLE_K = 0   # 0=保存全部视角；>0=每个 tar 仅保存前 K 张
NUM_WORKERS = max(1, cpu_count() // 2)  # 多进程并行数量

def to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint16:
        return (x.astype(np.uint32) // 257).astype(np.uint8)
    if x.dtype != np.uint8:
        return np.clip(x, 0, 255).astype(np.uint8)
    return x

def safe_save_rgb(np_rgb: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np_rgb).save(out_path)

def process_one_tar(tar_path: Path) -> str:
    uid = tar_path.stem
    out_dir = OUT_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        with tarfile.open(tar_path, "r") as tar:
            # 只挑 PNG，按名字排序（00000/00000.png …）
            pngs = sorted([m for m in tar.getmembers() if m.name.endswith(".png")], key=lambda m: m.name)
            if SAMPLE_K > 0:
                pngs = pngs[:SAMPLE_K]
            for m in pngs:
                # 输出文件名：最后两级拼接，如 00000/00000.png → 00000_00000.png
                fn = "_".join(m.name.split("/")[-2:])
                out_path = out_dir / fn
                if out_path.exists():
                    continue

                f = tar.extractfile(m)
                if f is None:
                    continue
                data = np.frombuffer(f.read(), np.uint8)
                img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # HxW x {3|4} 或 16-bit
                if img is None or img.ndim != 3:
                    continue

                img = to_uint8(img)
                if img.shape[2] == 4:
                    # BGRA → 白底 BGR
                    bgr = img[:, :, :3].astype(np.float32)
                    a   = (img[:, :, 3:4].astype(np.float32) / 255.0)
                    bgr = bgr * a + 255.0 * (1.0 - a)
                    bgr = bgr.astype(np.uint8)
                else:
                    bgr = img

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                safe_save_rgb(rgb, out_path)
                saved += 1
    except Exception as e:
        return f"[ERR] {uid}: {e}"
    return f"[OK] {uid}: saved {saved}"

def main():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tars = sorted(SRC_DIR.glob("*.tar"))
    if not tars:
        print(f"[WARN] no .tar found in {SRC_DIR}")
        return

    print(f"[INFO] total tars: {len(tars)}, sample_k={SAMPLE_K}, workers={NUM_WORKERS}")
    if NUM_WORKERS > 1:
        with Pool(NUM_WORKERS) as pool:
            for msg in tqdm(pool.imap_unordered(process_one_tar, tars), total=len(tars)):
                if msg:
                    print(msg)
    else:
        for tp in tqdm(tars):
            print(process_one_tar(tp))

    print(f"[DONE] previews at {OUT_DIR}")

if __name__ == "__main__":
    main()
