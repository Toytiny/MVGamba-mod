#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 在导入 pyrender 前强制用 EGL（无显示环境）
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys, json, csv, math, tarfile, shutil, tempfile, argparse, re, uuid
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 可选依赖（下载/渲染时才需要） ---
try:
    import objaverse
except Exception:
    objaverse = None
try:
    import trimesh, pyrender
except Exception:
    trimesh = pyrender = None


def try_resolve_mesh_path(u: str) -> Optional[Path]:
    """按 原样/32/36 三种形式尝试得到本地 mesh 路径。命中则返回 Path，否则 None。"""
    if objaverse is None:
        raise RuntimeError("请先安装 objaverse： pip install objaverse")

    tried = []
    s0 = u.strip()
    tried.append(s0)
    u_no = strip_dash(s0)
    if u_no not in tried:
        tried.append(u_no)
    try:
        if re.fullmatch(r"[0-9a-f]{32}", u_no):
            u_d = add_dash(u_no)
            if u_d not in tried:
                tried.append(u_d)
    except Exception:
        pass

    for cand in tried:
        meta = objaverse.load_objects({cand: None})
        p = meta.get(cand, None)
        if p:
            return Path(p)
    return None


def ensure_tar_for_uid(uid: str,
                       save_dir: Path,
                       img_size: int,
                       n_views: int,
                       cam_radius: float) -> bool:
    """为单个 uid 产出 <uid>.tar；若已存在且通过校验则直接返回 True。"""
    tar_path = save_dir / f"{uid}.tar"
    if tar_path.exists() and check_tar_integrity(tar_path):
        return True

    mesh_path = try_resolve_mesh_path(uid)
    if not mesh_path or not mesh_path.exists():
        # miss
        return False

    tp = render_and_pack_one(uid, mesh_path, save_dir,
                             img_size=img_size,
                             n_views=n_views,
                             cam_radius=cam_radius)
    return bool(tp and tp.exists() and check_tar_integrity(tp))

# ------------------------- 工具函数 -------------------------

def norm_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


def looks_like_uuid(s: str) -> bool:
    """支持 32 位(无横杠) 或 36 位(带横杠) 十六进制 UUID"""
    s = (s or "").strip().lower()
    return bool(
        re.fullmatch(r"[0-9a-f]{32}", s) or
        re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", s)
    )


def strip_dash(u: str) -> str:
    return (u or "").replace("-", "").lower()


def add_dash(u32: str) -> str:
    """32→36 标准 UUID 字符串"""
    return str(uuid.UUID(u32.lower()))


def read_uids(ids_path: Path) -> List[str]:
    """
    读取子集清单：
      - .json  : 列表或 [{'uid': ...}, ...]
      - .csv   : 自动识别 UUID 列（32/36），优先取第一列匹配的
      - .txt/.list : 每行一个 uid
    返回原样 UUID 字符串（不强制加横杠）
    """
    if ids_path.suffix == ".json":
        data = json.load(open(ids_path))
        raw = [x.get("uid", x) if isinstance(x, dict) else x for x in data]
        uids = [str(v).strip() for v in raw if looks_like_uuid(str(v))]

    elif ids_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(ids_path)
        candidate_col = None
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(200).tolist()
            if any(looks_like_uuid(v) for v in vals):
                candidate_col = c
                break
        if candidate_col is None:
            raise ValueError(
                f"No UUID-like column found in CSV: {ids_path}\nColumns={list(df.columns)}\n"
                f"Hint: 你的首列可能是内部编号（如 000-001），请使用包含 Objaverse UUID 的列。"
            )
        uids = [str(v).strip() for v in df[candidate_col].dropna().astype(str).tolist()
                if looks_like_uuid(str(v))]

    elif ids_path.suffix in [".txt", ".list"]:
        raw = [l.strip() for l in open(ids_path) if l.strip() and not l.startswith("#")]
        uids = [v for v in raw if looks_like_uuid(v)]

    else:
        raise ValueError(f"Unsupported ids file: {ids_path}")

    if not uids:
        raise ValueError(f"No valid UUID parsed from {ids_path}")
    return uids


def check_tar_integrity(tar_path: Path) -> bool:
    try:
        with tarfile.open(tar_path, "r") as tar:
            _ = tar.getmembers()
        return True
    except tarfile.TarError:
        return False


# ------------------------- 渲染相关 -------------------------

def look_at(cam_pos: np.ndarray,
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)) -> np.ndarray:
    """OpenGL 风格 c2w"""
    f = (target - cam_pos).astype(np.float32)
    f /= np.linalg.norm(f) + 1e-8
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-8
    u = np.cross(s, f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = s
    c2w[:3, 1] = u
    c2w[:3, 2] = f
    c2w[:3, 3] = cam_pos
    return c2w


def render_and_pack_one(uid: str,
                        mesh_path: Path,
                        save_dir: Path,
                        img_size: int = 512,
                        n_views: int = 40,
                        cam_radius: float = 2.0) -> Optional[Path]:
    """
    渲染 N 视角并打包为 <uid>.tar
    内部结构：<uid>/campos_512_v4/<vid>/<vid>.png 和 <vid>.json（含 x/y/z/origin）
    """
    if trimesh is None or pyrender is None:
        raise RuntimeError("请先安装 trimesh 和 pyrender： pip install trimesh pyrender")

    uid_last = uid  # 用规范化的 uuid 直接作为目录名
    tmp_root = Path(tempfile.mkdtemp())
    base = tmp_root / uid_last / "campos_512_v4"
    base.mkdir(parents=True, exist_ok=True)

    # 载入网格
    try:
        scene_trimesh = trimesh.load(mesh_path, force="mesh")
        mesh = (trimesh.util.concatenate([m for m in scene_trimesh.dump().geometries])
                if isinstance(scene_trimesh, trimesh.Scene) else scene_trimesh)
        try:
            mesh.apply_translation(-mesh.center_mass)
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] load mesh failed uid={uid} path={mesh_path}: {e}")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return None

    # 场景与光源
    scene = pyrender.Scene(ambient_light=[.3, .3, .3, 1.0])
    material = pyrender.MetallicRoughnessMaterial()
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(mesh_node)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light)

    # 相机与渲染器
    yfov = 2 * math.atan(1.0)  # ~90deg
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(img_size, img_size)

    elev = np.deg2rad(20.0)
    for i in range(n_views):
        az = 2 * math.pi * i / n_views
        cam_pos = np.array([cam_radius * math.cos(az) * math.cos(elev),
                            cam_radius * math.sin(elev),
                            cam_radius * math.sin(az) * math.cos(elev)], dtype=np.float32)
        c2w = look_at(cam_pos)

        cam_node = scene.add(camera, pose=c2w)
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)
        vdir = base / f"{i:05d}"
        vdir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(vdir / f"{i:05d}.png")
        meta = {
            "x": c2w[:3, 0].tolist(),
            "y": c2w[:3, 1].tolist(),
            "z": c2w[:3, 2].tolist(),
            "origin": c2w[:3, 3].tolist(),
        }
        json.dump(meta, open(vdir / f"{i:05d}.json", "w"))

    renderer.delete()

    # 打包
    save_dir.mkdir(parents=True, exist_ok=True)
    tar_path = save_dir / f"{uid}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for root, _, files in os.walk(tmp_root / uid_last):
            for fn in files:
                full = Path(root) / fn
                arc = str(full.relative_to(tmp_root))  # <uid>/campos_512_v4/...
                tar.add(full, arcname=arc)

    shutil.rmtree(tmp_root, ignore_errors=True)
    return tar_path


# ------------------------- 下载 Objaverse -------------------------

def download_objaverse(uids: List[str]) -> Dict[str, str]:
    """
    对每个 uid 三连尝试：
      1) 原样
      2) 去横杠(32)
      3) 32→36 加横杠
    命中即加入返回：{原始uid: 本地文件路径}
    """
    if objaverse is None:
        raise RuntimeError("请先安装 objaverse： pip install objaverse")

    result = {}
    for u in tqdm(uids, desc="download", ncols=100):
        tried = []
        s0 = u.strip()
        tried.append(s0)
        u_no = strip_dash(s0)
        if u_no not in tried:
            tried.append(u_no)
        try:
            if re.fullmatch(r"[0-9a-f]{32}", u_no):
                u_d = add_dash(u_no)
                if u_d not in tried:
                    tried.append(u_d)
        except Exception:
            pass

        found = False
        for cand in tried:
            meta = objaverse.load_objects({cand: None})
            path = meta.get(cand)
            if path is not None:
                result[u] = str(path)
                found = True
                break
        if not found:
            # 没找到就跳过（保留为空，后续渲染阶段会 miss）
            pass

    return result


# ------------------------- 主流程 -------------------------

def main():
    ap = argparse.ArgumentParser(description="Streamed download & render to tar for MVGamba.")
    ap.add_argument("--ids_path", type=str, default='/home/fangqiang.d/MVGamba/objaverse_filter/kiuisobj_v1_merged_80K.csv')
    ap.add_argument("--out_root", type=str, default='/data/mvgamba_gobj/')
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--n_views", type=int, default=40)
    ap.add_argument("--cam_radius", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=220, help="仅尝试前 N 个 UID（0=不限制）")
    ap.add_argument("--max_ok", type=int, default=0, help="成功生成的 .tar 数量达到该值即提前停止（0=不启用）")
    ap.add_argument("--use_v1_intersection", action="store_true")
    ap.add_argument("--filter_downloadable", action="store_true")
    ap.add_argument("--objaverse_cache", type=str, default="")
    args = ap.parse_args()

    ids_path = norm_path(args.ids_path)
    out_root = norm_path(args.out_root)
    savedata = out_root / "savedata"
    out_root.mkdir(parents=True, exist_ok=True)
    savedata.mkdir(parents=True, exist_ok=True)

    # 可选：设置 objaverse 缓存目录（大数据盘）
    if args.objaverse_cache and objaverse is not None:
        cache_dir = norm_path(args.objaverse_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        objaverse.BASE_PATH = str(cache_dir)
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

    # 读 UID
    subset_uids_raw = read_uids(ids_path)
    if args.limit and args.limit > 0:
        subset_uids_raw = subset_uids_raw[:args.limit]

    # 可选：与 v1 取交集、按 annotations 过滤
    uids_for_download = subset_uids_raw
    if args.use_v1_intersection:
        if objaverse is None:
            raise RuntimeError("use_v1_intersection 需要安装 objaverse")
        print("[INFO] loading v1 full UID set ...")
        v1_all = set(objaverse.load_uids())  # 32位无横杠
        inter = [u for u in uids_for_download if strip_dash(u) in v1_all]
        print(f"[INFO] intersection: {len(inter)} / {len(uids_for_download)}")
        uids_for_download = inter

    if args.filter_downloadable and objaverse is not None:
        print("[INFO] filtering by annotations (isDownloadable=True) ...")
        ann = objaverse.load_annotations([strip_dash(u) for u in uids_for_download])
        mask = {u for u, a in ann.items() if a.get("isDownloadable", True)}
        kept = [u for u in uids_for_download if strip_dash(u) in mask]
        print(f"[INFO] keep {len(kept)} / {len(uids_for_download)}")
        uids_for_download = kept

    # 流式：每个 uid 逐个保证 tar
    ok = 0; skip = 0; miss = 0; err = 0
    pbar = tqdm(uids_for_download, ncols=100, desc="stream")
    for uid in pbar:
        tar_path = savedata / f"{uid}.tar"
        try:
            if tar_path.exists() and check_tar_integrity(tar_path):
                skip += 1
                pbar.set_description(f"skip={skip} ok={ok} miss={miss}")
                # 检查是否已满足 max_ok（把已有的也算作已完成）
                if args.max_ok and (ok + skip) >= args.max_ok:
                    break
                continue

            done = ensure_tar_for_uid(uid, savedata, args.img_size, args.n_views, args.cam_radius)
            if done:
                ok += 1
                if args.max_ok and ok >= args.max_ok:
                    pbar.set_description(f"skip={skip} ok={ok} miss={miss}")
                    break
            else:
                miss += 1
            pbar.set_description(f"skip={skip} ok={ok} miss={miss}")
        except Exception as e:
            err += 1
            pbar.set_description(f"err={err} skip={skip} ok={ok} miss={miss}")
            print(f"[WARN] {uid}: {e}")

    print(f"[DONE] ok={ok}, skip={skip}, miss={miss}, err={err}, out={savedata}")


if __name__ == "__main__":
    main()
