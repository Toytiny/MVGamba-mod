#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_video.py — 将一组可视化图片合成为视频

用法示例：
# 1) 基于目录所有 PNG（智能数字排序），12fps，自动探测 ffmpeg 并优先使用：
python make_video.py \
  --input_dir results/mvgamba_man_mini/vis_val \
  --output results/mvgamba_man_mini/vis_val.mp4 \
  --fps 30

# 2) 只合成 bev_* 开头的图，强制尺寸为 1280x720：
python make_video.py \
  --input_dir results/mvgamba_man/vis_train \
  --glob "bev_*.png" \
  --output results/mvgamba_man/vis_train/bev.mp4 \
  --fps 30 --size 1280x720

# 3) 只取前 200 帧、每隔 2 张取一张（抽帧），反向合成：
python make_video.py \
  --input_dir results/mvgamba_man/vis_train \
  --output results/mvgamba_man/vis_train/vis_rev.mp4 \
  --end 200 --every 2 --reverse
"""

import argparse
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

def parse_args():
    p = argparse.ArgumentParser(description="Make a video from images")
    p.add_argument("--input_dir", type=str, required=True, help="图片所在目录")
    p.add_argument("--glob", type=str, default="*.png", help="图片通配符（默认 *.png）")
    p.add_argument("--output", type=str, required=True, help="输出视频路径（.mp4/.mov/.avi）")
    p.add_argument("--fps", type=int, default=12, help="帧率（默认 12）")
    p.add_argument("--size", type=str, default="", help='强制输出分辨率，格式 "WIDTHxHEIGHT"，如 1280x720；留空则使用首帧尺寸')
    p.add_argument("--start", type=int, default=0, help="起始索引（含），默认 0")
    p.add_argument("--end", type=int, default=-1, help="结束索引（不含），默认 -1 表示到最后")
    p.add_argument("--every", type=int, default=1, help="抽帧步长（每 N 张取 1 张），默认 1")
    p.add_argument("--reverse", action="store_true", help="是否反向合成")
    p.add_argument("--codec", type=str, default="mp4v", help="OpenCV 模式使用的 fourcc（默认 mp4v，可尝试 H264/avc1）")
    p.add_argument("--quality", type=int, default=18, help="ffmpeg 模式 CRF 质量（0~51，数值越小质量越高，默认 18）")
    p.add_argument("--overwrite", action="store_true", help="若输出已存在则覆盖")
    return p.parse_args()

def numeric_key(s: str):
    """
    用于智能数字排序的 key：把字符串中的数字块转换为整数对齐比较。
    例如：img_2.png, img_10.png 会按 2 < 10 排序。
    """
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def collect_images(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(input_dir.glob(pattern), key=lambda p: numeric_key(p.name))
    # 再过滤一次扩展名（更稳）
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [f for f in files if f.suffix.lower() in exts]
    return files

def slice_select(files: List[Path], start: int, end: int, every: int, reverse: bool) -> List[Path]:
    n = len(files)
    if n == 0:
        return files
    if end < 0 or end > n:
        end = n
    files = files[start:end:every]
    if reverse:
        files = list(reversed(files))
    return files

def parse_size(size_str: str) -> Optional[Tuple[int, int]]:
    if not size_str:
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", size_str.strip().lower())
    if not m:
        raise ValueError('size 参数格式应为 "WIDTHxHEIGHT"，如 1280x720')
    return int(m.group(1)), int(m.group(2))

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def make_with_ffmpeg(files: List[Path], output: Path, fps: int, size: Optional[Tuple[int,int]], crf: int, overwrite: bool):
    """
    用 ffmpeg 合成：通过 -f concat 从临时文件列表读取，避免按名称顺序问题。
    """
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output} (use --overwrite to replace)")

    # 写一个 concat list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        for p in files:
            # ffmpeg concat 需要转义或用单引号；用绝对路径更稳
            tf.write(f"file '{p.resolve().as_posix()}'\n")
        list_file = tf.name

    vf_chain = []
    if size is not None:
        w, h = size
        # force scale with even dims to be safe for codecs
        vf_chain.append(f"scale={w}:{h}:flags=bicubic")
    vf = ",".join(vf_chain) if vf_chain else "null"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-r", str(fps),           # 输入帧率
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",    # 兼容多数播放器
        "-crf", str(crf),
    ]
    if overwrite:
        cmd.insert(1, "-y")
    cmd.append(str(output))

    try:
        subprocess.run(cmd, check=True)
        print(f"[ffmpeg] Done → {output}")
    finally:
        try:
            os.remove(list_file)
        except Exception:
            pass

def make_with_opencv(files: List[Path], output: Path, fps: int, size: Optional[Tuple[int,int]], fourcc: str, overwrite: bool):
    import cv2
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output} (use --overwrite to replace)")

    # 读首帧决定尺寸（若未显式指定）
    first = cv2.imdecode(np.fromfile(str(files[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first image: {files[0]}")
    if size is None:
        h, w = first.shape[:2]
    else:
        w, h = size
        first = cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA)

    # OpenCV 视频编码器
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    vw = cv2.VideoWriter(str(output), fourcc_code, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter with codec={fourcc}")

    try:
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="[OpenCV] Writing", ncols=80)
        except Exception:
            iterator = files

        for p in iterator:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] skip unreadable: {p}")
                continue
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            vw.write(img)
    finally:
        vw.release()
    print(f"[opencv] Done → {output}")

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    size = parse_size(args.size)

    files = collect_images(input_dir, args.glob)
    if not files:
        print(f"[error] no images found in {input_dir} with pattern '{args.glob}'")
        sys.exit(1)

    files = slice_select(files, args.start, args.end, args.every, args.reverse)
    if not files:
        print("[error] slice produced empty file list")
        sys.exit(1)

    print(f"[info] {len(files)} frames selected from {input_dir} (pattern={args.glob})")
    print(f"[info] fps={args.fps} size={'auto' if size is None else size} backend={'ffmpeg' if have_ffmpeg() else 'opencv'}")

    if have_ffmpeg():
        make_with_ffmpeg(files, output, args.fps, size, args.quality, args.overwrite)
    else:
        try:
            import cv2  # noqa
        except Exception:
            print("[error] neither ffmpeg nor opencv is available. Please install one of them.")
            sys.exit(1)
        make_with_opencv(files, output, args.fps, size, args.codec, args.overwrite)

if __name__ == "__main__":
    import numpy as np  # for OpenCV imdecode path with unicode
    main()
