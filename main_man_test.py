import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; needed for 3D projection

from core.options import AllConfigs, Options
from core.mvgamba_models2 import MVGamba2 as MVGamba

# === 可选：如果你用 accelerate/多卡训练，单卡测试更简单 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------- Metrics (aligned with training loss) ----------
@torch.no_grad()
def chamfer_l2(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """
    与训练一致：用 torch.cdist 的欧氏距离（非平方），再做双向均值。
    pred_pts: [B,K,3], gt_pts: [B,P,3]
    返回标量（batch 平均后的 Chamfer-L2）
    """
    D = torch.cdist(pred_pts, gt_pts)          # [B,K,P]
    d1 = D.min(dim=2).values.mean(dim=1)       # pred->gt
    d2 = D.min(dim=1).values.mean(dim=1)       # gt->pred
    return (d1 + d2).mean()


@torch.no_grad()
def attr_mae_nn(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                pred_attr: torch.Tensor, gt_attr: torch.Tensor) -> torch.Tensor:
    """
    与训练一致：先用 torch.cdist 找 pred->gt 最近邻，再对齐属性做 L1。
    pred_attr: [B,K,1], gt_attr: [B,P,1]
    返回 batch 平均 MAE
    """
    D = torch.cdist(pred_pts, gt_pts)          # [B,K,P]
    nn_idx = D.argmin(dim=2)                   # [B,K]
    idx = nn_idx.unsqueeze(-1).expand(-1, -1, 1)        # [B,K,1]
    nn_gt_attr = torch.gather(gt_attr, dim=1, index=idx)  # [B,K,1]
    mae = (pred_attr - nn_gt_attr).abs().mean(dim=(1,2))  # [B]
    return mae.mean()


# ---------- 工具：统一提取置信度 ----------
def _extract_pred_conf(out: dict) -> torch.Tensor:
    """
    返回 [B,K,1] 的置信度张量：
    1) 若 pred['points'] 最后一维 >= 6，取第 6 维为 conf；
    2) 否则若 pred['conf'] 存在，兼容 [B,K] / [B,K,1]；
    3) 都没有则返回全 1。
    """
    pts = out['pred']['points']  # [B,K,3/6]
    if pts.dim() == 3 and pts.shape[-1] >= 6:
        return pts[..., 5:6]
    conf = out['pred'].get('conf', None)
    if conf is None:
        B, K = pts.shape[0], pts.shape[1]
        return torch.ones(B, K, 1, device=pts.device, dtype=pts.dtype)
    if conf.dim() == 2:
        conf = conf.unsqueeze(-1)
    return conf


# ---------- 可视化：多视角输入 ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

# 统一固定颜色范围（RCS或Doppler）
ATTR_VMIN, ATTR_VMAX = -25.0, 25.0


def _best_denorm(cchw: np.ndarray) -> np.ndarray:
    """
    输入: [C,H,W] float32
      - C==1: 灰度 -> 复用成3通道
      - C>=3: 仅取前3个通道做可视化（例如 C=9: RGB+Plücker）
    返回: [H,W,3] uint8
    """
    c, h, w = cchw.shape
    x = cchw.copy()

    if c == 1:
        img01 = (x[0] - x[0].min()) / (x[0].ptp() + 1e-8)
        img3 = np.stack([img01, img01, img01], axis=-1)
        return _to_uint8(img3)

    # C >= 3: 只取前三通道可视化（RGB）
    x = x[:3, :, :]

    # 候选1：已在[0,1]
    cand1 = x.transpose(1, 2, 0)
    score1 = float(((cand1 >= 0) & (cand1 <= 1)).mean())

    # 候选2：[-1,1] -> [0,1]
    cand2 = ((x + 1.0) * 0.5).transpose(1, 2, 0)
    score2 = float(((cand2 >= 0) & (cand2 <= 1)).mean())

    # 候选3：ImageNet 反归一化
    cand3 = (x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN)
    score3 = float(((cand3 >= 0) & (cand3 <= 1)).mean())

    cands = [cand1, cand2, cand3]
    scores = [score1, score2, score3]
    best = cands[int(np.argmax(scores))]
    return _to_uint8(best)

from matplotlib import gridspec

def _plucker_global_range(mviews_np):
    """
    mviews_np: [V,C,H,W] numpy, C>=9 时取后6通道
    返回: (mins[6], maxs[6]) 供统一色域
    """
    V, C, H, W = mviews_np.shape
    assert C >= 6, "expect at least 6 channels for plucker"
    p = mviews_np[:, -6:, :, :]  # [V,6,H,W]
    p = p.reshape(V, 6, -1)
    mins = p.min(axis=2).min(axis=0)  # [6]
    maxs = p.max(axis=2).max(axis=0)  # [6]
    # 为零中心的对称色域，避免偏色：取对称极值
    absmax = np.maximum(np.abs(mins), np.abs(maxs))
    return -absmax, absmax

def save_panel_inputs_and_bev(
    mviews: torch.Tensor,               # [V,C,H,W]
    p_pred: np.ndarray, p_gt: np.ndarray,   # [N,3], [M,3]
    c_pred: np.ndarray|None, c_gt: np.ndarray|None,
    save_path: str,
    title_suffix: str = "",
    max_cols: int = 4,
    s_pred: int = 3, s_gt: int = 3,
    alpha_pred: float = 0.8, alpha_gt: float = 0.6,
    # ==== 颜色控制（仅影响 Pred 面板）====
    pred_vmin: float | None = None,
    pred_vmax: float | None = None,
    pred_cmap: str = 'seismic',
    pred_cbar_label: str = 'attr',
):
    """
    单张面板：上=多视角RGB；中=对应Plücker(6通道)；下=Pred/GT 的 BEV。
    """
    V, C, H, W = mviews.shape
    ncols_top = min(V, max_cols)
    nrows_top = int(np.ceil(V / ncols_top))
    has_plucker = (C >= 9)

    # 预处理多视角输入（只取前三通道可视化）
    imgs = []
    with torch.no_grad():
        mv = mviews.detach().cpu().float().numpy()  # [V,C,H,W]
        for v in range(V):
            imgs.append(_best_denorm(mv[v]))  # [H,W,3] uint8
        if has_plucker:
            pmins, pmaxs = _plucker_global_range(mv)  # 各通道统一色域（对称零中心）
        else:
            pmins = pmaxs = None

    # 统一 BEV 范围
    x_pred, y_pred = p_pred[:, 0], p_pred[:, 1]
    x_gt,   y_gt   = p_gt[:, 0],   p_gt[:, 1]
    x_all = np.concatenate([x_pred, x_gt]); y_all = np.concatenate([y_pred, y_gt])
    margin = 0.02
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    dx, dy = x_max - x_min, y_max - y_min
    x_pad, y_pad = dx * margin, dy * margin
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # 画布布局：上 nrows_top 行（RGB），中（如有） nrows_top 行（Plücker），最后 1 行（BEV 两列）
    total_rows = nrows_top + (nrows_top if has_plucker else 0) + 1
    fig = plt.figure(figsize=(4*ncols_top, 3.0*nrows_top + (2.6*nrows_top if has_plucker else 0) + 5.2))
    gs = gridspec.GridSpec(total_rows, ncols_top, figure=fig)

    # ---- 上：多视角 RGB ----
    k = 0
    for r in range(nrows_top):
        for c_ in range(ncols_top):
            ax = fig.add_subplot(gs[r, c_])
            if k < V:
                ax.imshow(imgs[k])
                ax.set_title(f"View {k} (C={C})", fontsize=10)
                k += 1
            ax.axis('off')

    # ---- 中：Plücker 6通道（每格 2×3 子网格）----
    if has_plucker:
        labels = ["m_x","m_y","m_z","d_x","d_y","d_z"]
        k = 0
        for r in range(nrows_top, nrows_top*2):
            for c_ in range(ncols_top):
                ax_parent = fig.add_subplot(gs[r, c_])
                ax_parent.axis('off')
                if k < V:
                    sub = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[r, c_], wspace=0.05, hspace=0.05)
                    for i in range(6):
                        axp = fig.add_subplot(sub[i//3, i%3])
                        im = axp.imshow(
                            mv[k, -6 + i],  # 第 k 个视角的第 i 个 Plücker 通道
                            cmap='seismic',
                            vmin=pmins[i], vmax=pmaxs[i]
                        )
                        axp.set_xticks([]); axp.set_yticks([])
                        axp.set_title(labels[i], fontsize=8)
                        # 只在最后一幅放 colorbar，避免太拥挤
                        if i == 5:
                            cbar = fig.colorbar(im, ax=axp, fraction=0.046, pad=0.02)
                            cbar.ax.tick_params(labelsize=7)
                    ax_parent.set_title(f"View {k} — Plücker", fontsize=10)
                    k += 1

    # ---- 下：BEV 左右对比 ----
    last_row = total_rows - 1
    left_span = slice(0, ncols_top // 2 if ncols_top >= 2 else 1)
    right_span = slice(ncols_top // 2 if ncols_top >= 2 else 1, ncols_top)

    axL = fig.add_subplot(gs[last_row, left_span])
    if c_pred is None:
        axL.scatter(x_pred, y_pred, s=s_pred, alpha=alpha_pred)
    else:
        sc1 = axL.scatter(
            x_pred, y_pred,
            c=c_pred.reshape(-1), s=s_pred, alpha=alpha_pred,
            cmap=pred_cmap, vmin=pred_vmin, vmax=pred_vmax   # 仅 Pred 受控
        )
        cbar1 = fig.colorbar(sc1, ax=axL, shrink=0.7, pad=0.02)
        cbar1.set_label(pred_cbar_label, rotation=90)

    axL.set_title('Pred (BEV)' + title_suffix)
    axL.set_xlabel('X'); axL.set_ylabel('Y')
    axL.set_aspect('equal', adjustable='box'); axL.set_xlim(xlim); axL.set_ylim(ylim)
    axL.grid(True, ls='--', alpha=0.3)
    
    axR = fig.add_subplot(gs[last_row, right_span])
    if c_gt is None:
        axR.scatter(x_gt, y_gt, s=s_gt, alpha=alpha_gt)
    else:
        sc2 = axR.scatter(
            x_gt, y_gt,
            c=c_gt.reshape(-1), s=s_gt, alpha=alpha_gt,
            cmap='seismic', vmin=ATTR_VMIN, vmax=ATTR_VMAX   # GT 保持原配置
        )
        cbar2 = fig.colorbar(sc2, ax=axR, shrink=0.7, pad=0.02)
        cbar2.set_label('attr', rotation=90)

    axR.set_title('GT (BEV)' + title_suffix)
    axR.set_xlabel('X'); axR.set_ylabel('Y')
    axR.set_aspect('equal', adjustable='box'); axR.set_xlim(xlim); axR.set_ylim(ylim)
    axR.grid(True, ls='--', alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[save] {save_path}")


# ---------- 主测试流程 ----------
@torch.no_grad()
def run_test(ckpt_path=None,
             data_mode='s3',
             eval_split='mini_val',   # 或 full val
             batch_size=4,
             num_workers=4,
             out_dir="results/vis_eval",
             max_batches=100,           # 只跑前 N 个 batch 做 sanity check；设为 None 跑完整集
             color_by='rcs'           # 'rcs' | 'doppler' | 'conf' | None
             ):
    # 1) 配置 & 模型
    opt = Options()  # 用默认配置；如需自定义可改
    opt.data_mode = data_mode
    opt.input_size = getattr(opt, "input_size", 448)
    opt.batch_size = batch_size

    model = MVGamba(opt).to(DEVICE)
    model.eval()
    if ckpt_path is not None:
        print(f"[load] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)

    # 2) 数据
    if data_mode == 's3':
        from core.truckscenes_dataset import MANTruckscenesDataset as Dataset
    else:
        raise NotImplementedError

    dataset = Dataset(opt, training=False, eval_split=eval_split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    os.makedirs(out_dir, exist_ok=True)

    # 3) 遍历 & 计算指标
    meter = {"cd": [], "rcs_mae": [], "dop_mae": []}

    # ---------- 第 1 遍：统计全局 vmin/vmax ----------
    global_vmin, global_vmax = None, None
    need_color = (color_by in ['rcs', 'doppler', 'conf'])

    if need_color:
        if color_by == 'conf':
            # 置信度固定 [0,1]，无需遍历
            global_vmin, global_vmax = 0.0, 1.0
            print(f"[global color range] conf: vmin={global_vmin}, vmax={global_vmax}")
        else:
            for bi, data in enumerate(loader):
                def to_dev(x):
                    if isinstance(x, dict):
                        return {k: to_dev(v) for k, v in x.items()}
                    elif torch.is_tensor(x):
                        return x.to(DEVICE, non_blocking=True)
                    else:
                        return x
                data = to_dev(data)
                out = model(data, epoch=0, step_ratio=1.0, vis=1)

                # 取出属性（pred/gt 都参与范围）
                if color_by == 'rcs':
                    pred_attr = out['pred']['rcs']      # [B,K,1]
                    gt_attr   = out['gt']['rcs']        # [B,P,1]
                else:  # 'doppler'
                    pred_attr = out['pred']['doppler']  # [B,K,1]
                    gt_attr   = out['gt']['doppler']    # [B,P,1]

                # 转到 numpy 并打平
                c_pred = pred_attr.detach().cpu().numpy().reshape(-1)
                c_gt   = gt_attr.detach().cpu().numpy().reshape(-1)

                # 更新全局范围（跳过 NaN）
                c_all = np.concatenate([c_pred, c_gt])
                c_all = c_all[~np.isnan(c_all)]
                if c_all.size > 0:
                    vmin_batch, vmax_batch = float(c_all.min()), float(c_all.max())
                    global_vmin = vmin_batch if global_vmin is None else min(global_vmin, vmin_batch)
                    global_vmax = vmax_batch if global_vmax is None else max(global_vmax, vmax_batch)

            print(f"[global color range] {color_by}: vmin={global_vmin}, vmax={global_vmax}")

    # ---------- 第 2 遍：计算指标 + 全量出图 ----------
    # 重新建一个 loader（上面的 loader 可能已经被消费完）
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    sample_counter = 0  # 全局样本编号，方便命名
    for bi, data in enumerate(loader):
        def to_dev(x):
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            elif torch.is_tensor(x):
                return x.to(DEVICE, non_blocking=True)
            else:
                return x
        data = to_dev(data)

        out = model(data, epoch=0, step_ratio=1.0, vis=1)

        # ==== confidence 可视化筛选参数（可放在 for bi 外面）====
        CONF_THRESH = float(getattr(opt, "conf_thresh", 0))   # 置信度阈值
        MIN_KEEP    = int(getattr(opt, "conf_min_keep", 1296))  # 阈值太高时，至少保留这么多点
        K_EVAL_CAP  = int(getattr(opt, "K_eval", 8192))         # 进一步上限，避免图太密/太慢

        # ----- 取点云 -----
        pred_pts = out['pred']['points']      # [B,K,3] 或 [B,K,6]
        pred_rcs = out['pred']['rcs']         # [B,K,1]
        pred_dop = out['pred']['doppler']     # [B,K,1]
        gt_pts   = out['gt']['points']        # [B,P,3]
        gt_rcs   = out['gt']['rcs']           # [B,P,1]
        gt_dop   = out['gt']['doppler']       # [B,P,1]

        # ====== NEW: batch 级 XY 统计（pred / gt）======
        def _xy_stats(t: torch.Tensor):
            # t: [*,3]
            x = t[..., 0].reshape(-1).float()
            y = t[..., 1].reshape(-1).float()
            return (x.min().item(), x.max().item(), x.mean().item(),
                    y.min().item(), y.max().item(), y.mean().item())

        b_pred_xy = _xy_stats(pred_pts)  # (xmin, xmax, xmean, ymin, ymax, ymean)
        b_gt_xy   = _xy_stats(gt_pts)

        print(
            "[batch {:03d}] pred_x[min,max,mean]=[{:.2f},{:.2f},{:.2f}]  "
            "pred_y[min,max,mean]=[{:.2f},{:.2f},{:.2f}]  |  "
            "gt_x[min,max,mean]=[{:.2f},{:.2f},{:.2f}]  "
            "gt_y[min,max,mean]=[{:.2f},{:.2f},{:.2f}]".format(
                bi,
                b_pred_xy[0], b_pred_xy[1], b_pred_xy[2],
                b_pred_xy[3], b_pred_xy[4], b_pred_xy[5],
                b_gt_xy[0],   b_gt_xy[1],   b_gt_xy[2],
                b_gt_xy[3],   b_gt_xy[4],   b_gt_xy[5],
            )
        )

        # ----- metrics（batch 级，与训练完全一致）-----
        cd = chamfer_l2(pred_pts, gt_pts).item()
        rcs_mae = attr_mae_nn(pred_pts, gt_pts, pred_rcs, gt_rcs).item()
        dop_mae = attr_mae_nn(pred_pts, gt_pts, pred_dop, gt_dop).item()

        # 与训练 loss 同权重的评估标量，便于一眼对齐：
        eval_loss = cd + 0.1 * rcs_mae + 0.2 * dop_mae

        meter["cd"].append(cd)
        meter["rcs_mae"].append(rcs_mae)
        meter["dop_mae"].append(dop_mae)

        print(f"[batch {bi}] cd={cd:.4f}  rcs_mae={rcs_mae:.4f}  dop_mae={dop_mae:.4f}  eval_loss={eval_loss:.4f}")

        # 遍历本 batch 里的每个样本：保存组合面板
        B = pred_pts.shape[0]
        input_imgs = data['input']['images']  # [B,V,C,H,W]

        # 统一提取本 batch 的置信度
        pred_conf_b = _extract_pred_conf(out)  # [B,K,1]

        for b in range(B):
            # ===== 解析输出（兼容 5/6 维两种结构）=====
            pts_full = out['pred']['points'][b]        # shape 可能是 [K,3] 或 [K,6]
            if pts_full.shape[-1] >= 3:
                p_xyz = pts_full[..., :3]              # [K,3]
            else:
                raise ValueError("pred['points'] last dim < 3")

            p_rcs  = out['pred']['rcs'][b].reshape(-1, 1)       # [K,1]
            p_dopp = out['pred']['doppler'][b].reshape(-1, 1)   # [K,1]
            p_conf = pred_conf_b[b]                             # [K,1]

            # ===== 根据阈值筛选（不足 MIN_KEEP 用 Top-K 回填；再截断到 K_EVAL_CAP）=====
            conf_flat = p_conf.squeeze(-1)
            mask = conf_flat >= CONF_THRESH
            keep_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)

            if keep_idx.numel() < min(MIN_KEEP, p_xyz.shape[0]):
                # 用 Top-K(conf) 进行回填
                k_fill = min(max(MIN_KEEP, keep_idx.numel()), p_xyz.shape[0])
                topk_idx = torch.topk(conf_flat, k=k_fill, dim=0).indices
                keep_idx = topk_idx

            if keep_idx.numel() > K_EVAL_CAP:
                # 截断到最多 K_EVAL_CAP（仍按 conf 排序）
                k_cap_idx = torch.topk(conf_flat[keep_idx], k=K_EVAL_CAP, dim=0).indices
                keep_idx = keep_idx[k_cap_idx]

            # 实际筛选出的预测
            p_pred_xyz = p_xyz[keep_idx]        # [K_sel,3]
            p_pred_rcs = p_rcs[keep_idx]        # [K_sel,1]
            p_pred_dop = p_dopp[keep_idx]       # [K_sel,1]
            p_pred_conf= p_conf[keep_idx]       # [K_sel,1]

            # ===== 取 GT =====
            p_gt   = gt_pts[b].detach().cpu().numpy()

            # ===== 打印一些统计（可选）=====
            kept = int(keep_idx.numel())
            thr  = float(CONF_THRESH)
            cmean= float(p_pred_conf.mean().item())
            print(f"[sample {sample_counter:06d}] conf>= {thr:.2f} -> keep {kept} / {p_xyz.shape[0]} (mean={cmean:.3f})")

            # ===== 可视化分支 =====
            if color_by == 'conf':
                # 用置信度上色：范围 [0,1]，色条标注为 'conf'
                c_pred = p_pred_conf.detach().cpu().numpy().reshape(-1)
                c_gt   = None
                title_suffix = " (Conf)"
                pred_vmin, pred_vmax = 0.0, 1.0
                pred_cbar_label = 'conf'
            elif color_by == 'rcs':
                c_pred = p_pred_rcs.detach().cpu().numpy().reshape(-1)
                c_gt   = gt_rcs[b].detach().cpu().numpy().reshape(-1)
                title_suffix = " (RCS)"
                pred_vmin = pred_vmax = None
                pred_cbar_label = 'attr'
            elif color_by == 'doppler':
                c_pred = p_pred_dop.detach().cpu().numpy().reshape(-1)
                c_gt   = gt_dop[b].detach().cpu().numpy().reshape(-1)
                title_suffix = " (Doppler)"
                pred_vmin = pred_vmax = None
                pred_cbar_label = 'attr'
            else:
                c_pred = None; c_gt = None; title_suffix = ""
                pred_vmin = pred_vmax = None
                pred_cbar_label = 'attr'

            save_png_panel = os.path.join(out_dir, f"panel_sample{sample_counter:06d}.png")
            save_panel_inputs_and_bev(
                mviews=input_imgs[b],
                p_pred=p_pred_xyz.detach().cpu().numpy(),
                p_gt=gt_pts[b].detach().cpu().numpy(),
                c_pred=c_pred, c_gt=c_gt,
                save_path=save_png_panel,
                title_suffix=title_suffix,
                max_cols=4,
                s_pred=3, s_gt=3,
                alpha_pred=0.8, alpha_gt=0.6,
                # 置信度/属性的色域与标签
                pred_vmin=pred_vmin, pred_vmax=pred_vmax,
                pred_cmap='viridis' if color_by=='conf' else 'seismic',
                pred_cbar_label=pred_cbar_label,
            )

            # （可选）顺便把 conf 直方图存一张，调阈值很有用
            hist_path = os.path.join(out_dir, f"conf_hist_{sample_counter:06d}.png")
            try:
                plt.figure(figsize=(4,3))
                plt.hist(conf_flat.detach().cpu().numpy(), bins=50, range=(0,1))
                plt.title(f"Conf Histogram (sample {sample_counter:06d})")
                plt.xlabel("confidence"); plt.ylabel("count")
                plt.tight_layout(); plt.savefig(hist_path, dpi=160); plt.close()
            except Exception as e:
                print(f"[warn] save conf hist failed: {e}")

            sample_counter += 1

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    # 4) 汇总
    def _mean(x): return float(np.mean(x)) if len(x) > 0 else float("nan")
    summary = {
        "cd_mean": _mean(meter["cd"]),
        "rcs_mae_mean": _mean(meter["rcs_mae"]),
        "dop_mae_mean": _mean(meter["dop_mae"]),
        "eval_loss_mean": _mean(meter["cd"]) + 0.1*_mean(meter["rcs_mae"]) + 0.2*_mean(meter["dop_mae"]),
        "num_batches": len(meter["cd"]),
    }

    print("\n=== Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # 写一个摘要到文件
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return summary


if __name__ == "__main__":
    # 示例：根据需要替换 ckpt_path/eval_split 等
    run_test(
        ckpt_path="/home/fangqiang.d/MVGamba/results/mvgamba_man_1_wcd/checkpoint_ep2600.pth",
        data_mode='s3',
        eval_split='mini_train',
        batch_size=64,
        num_workers=8,
        out_dir="results/mvgamba_man_1_wcd/vis_train",
        max_batches=1000,
        color_by='conf',               # 'rcs' or 'doppler' or 'conf' or None
    )
