"""
resnet_val_main_0209.py
-----------------------------------------------------------------------------
用途
- 加载训练阶段保存的 checkpoint（best.pt），在指定 NPZ 上做推理，导出与训练一致口径的 CSV；
- 可选：同时调用传统物理闭式反解（Python 版 solve_physics_batch）输出 “_phy” 结果用于对比；
- 若 NPZ 含 y_true，则计算误差指标（MSE/MAE/RMSE，含 alpha/beta 分维度）；
- 额外：按 (Delta_alpha_true, Beta_true) 精确分组统计组内 std（衡量“同参数不同采样”的预测波动）。

输入
- ckpt_path : 训练输出的 best.pt（必须包含 y_mean/y_std/length/norm_x 等）
- val_npz  : 输入数据 NPZ
  - 必须包含 x；可选包含 y（真实尺度）
  - 若启用传统方法：必须包含 lambda_nm；可选包含 I0（否则默认 0.01）
- out_dir   : 输出目录

与训练保持一致的处理
- x 预处理：若 ckpt['norm_x']=True，则每样本沿 T 做 min-max -> [0,1]
- y 反变换：脚本使用 ckpt['y_mean'] 与 ckpt['y_std'] 做反标准化，得到真实尺度 y_pred_real
  - 注意：脚本内保留了 y_mode/y_scale 的旧逻辑注释，但当前实现“强制按 standardize 反归一化”。

传统方法对比（可选）
- 依赖：original_matlab2python_0209.solve_physics_batch
- 注意：传统方法必须使用“未做 min-max 的原始 x_raw”，否则幅值关系被破坏。
- 传统方法输出可能出现 NaN（无解）；脚本会统计 valid 比例，并只在 valid 样本上计算误差。

主要函数（功能/输入输出）
- load_npz_x_y(npz_path, length) -> (x:Tensor[N,1,L], y_true:Tensor[N,2] 或 None)
- minmax_per_sample_T(x[N,1,L]) -> x_norm[N,1,L]
- infer(model, loader, device) -> (y_pred_norm:Tensor[N,2], y_true_raw 或 None)
- inverse_y_to_real(y_pred_norm, ckpt) -> y_pred_real:Tensor[N,2]
- compute_metrics(y_true_real, y_pred_real) -> dict（总体/分维度 mse/mae/rmse）
- export_csv_same_format(csv_path, y_true, y_pred, y_pred_phy=None)
  - 导出列固定，并额外含 sample_idx；若无 y_true 则 true/diff 填 NaN
- group_std_by_exact_ytrue(csv_path, out_csv=None) -> (stats_df, summary_dict)
  - 按 (Delta_alpha_true, Beta_true) 分组统计预测均值/标准差，并打印汇总

输出（写入 out_dir）
- best_epoch_XXX_y_true_y_pred.csv
  - 含网络预测列；若启用传统方法，还会追加 *_phy 列
- groupwise_std.csv（仅当 npz 含 y_true 时生成）
- 终端打印：ckpt 信息、网络指标、传统方法 valid% 与指标等

依赖
- torch, numpy, pandas, tqdm
- resnet1d.py（提供 resnet50_1d）
- original_matlab2python_0209.py（提供 solve_physics_batch，用于传统方法对比）
-----------------------------------------------------------------------------
"""

import os
import numpy as np
from tqdm import tqdm

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from resnet1d import resnet50_1d

# 传统方法（Matlab->Python 版）：保持原公式，只做 I/O 适配
from original_matlab2python_0209 import solve_physics_batch

import pandas as pd

def group_std_by_exact_ytrue(csv_path: str, out_csv: str = None):
    df = pd.read_csv(csv_path)

    a_true = "Delta_alpha_true"
    b_true = "Beta_true"
    a_pred = "Delta_alpha_pred"
    b_pred = "Beta_pred"
    a_pred_phy = "Delta_alpha_pred_phy"
    b_pred_phy = "Beta_pred_phy"

    has_phy = (a_pred_phy in df.columns) and (b_pred_phy in df.columns)

    grp = df.groupby([a_true, b_true], sort=False)

    agg_dict = {
        "n": (a_true, "size"),
        "alpha_true": (a_true, "first"),
        "beta_true": (b_true, "first"),
        "alpha_pred_mean": (a_pred, "mean"),
        "beta_pred_mean": (b_pred, "mean"),
        "alpha_pred_std": (a_pred, "std"),
        "beta_pred_std": (b_pred, "std"),
    }

    if has_phy:
        agg_dict.update(
            {
                "alpha_pred_phy_mean": (a_pred_phy, "mean"),
                "beta_pred_phy_mean": (b_pred_phy, "mean"),
                "alpha_pred_phy_std": (a_pred_phy, "std"),
                "beta_pred_phy_std": (b_pred_phy, "std"),
            }
        )

    stats = grp.agg(**agg_dict).reset_index(drop=True)

    # 组内样本数=1 时 std 为 NaN，这种组没有“组内波动”意义；通常置 0
    # 组内样本数=1 时 std 为 NaN，这种组没有“组内波动”意义；通常置 0
    stats["alpha_pred_std"] = stats["alpha_pred_std"].fillna(0.0)
    stats["beta_pred_std"]  = stats["beta_pred_std"].fillna(0.0)

    if has_phy:
        # 注意：传统方法可能出现“整组都无解（全 NaN）”，这类组的 std 也会是 NaN。
        # - n==1 的组：std NaN -> 0（同上）
        # - 全 NaN 的组：std NaN 保持 NaN（用于统计 fail_rate）
        is_small_n = (stats["n"] <= 1)
        stats.loc[is_small_n, "alpha_pred_phy_std"] = stats.loc[is_small_n, "alpha_pred_phy_std"].fillna(0.0)
        stats.loc[is_small_n, "beta_pred_phy_std"]  = stats.loc[is_small_n, "beta_pred_phy_std"].fillna(0.0)

    # 汇总（不加权/加权）
    n = stats["n"].to_numpy()
    sa = stats["alpha_pred_std"].to_numpy()
    sb = stats["beta_pred_std"].to_numpy()

    summary = {
        "num_groups": int(len(stats)),
        "mean_std_alpha": float(np.mean(sa)),
        "weighted_mean_std_alpha": float(np.sum(n * sa) / np.sum(n)),
        "mean_std_beta": float(np.mean(sb)),
        "weighted_mean_std_beta": float(np.sum(n * sb) / np.sum(n)),
    }

    if has_phy:
        sap = stats["alpha_pred_phy_std"].to_numpy()
        sbp = stats["beta_pred_phy_std"].to_numpy()
        summary.update(
            {
                "mean_std_alpha_phy": float(np.nanmean(sap)),
                "weighted_mean_std_alpha_phy": float(np.nansum(n * sap) / np.sum(n)),
                "mean_std_beta_phy": float(np.nanmean(sbp)),
                "weighted_mean_std_beta_phy": float(np.nansum(n * sbp) / np.sum(n)),
            }
        )

        # 额外：统计“全 NaN 组”比例（用于解释 fail_rate）
        grp_all_nan = grp[[a_pred_phy, b_pred_phy]].apply(lambda g: (~np.isfinite(g.to_numpy())).all())
        # grp_all_nan 的 index 与 group key 对应，这里只要比例即可
        summary["group_all_nan_rate_phy"] = float(grp_all_nan.mean())

    print("[Groupwise std by exact y_true]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if out_csv is not None:
        stats.to_csv(out_csv, index=False)
        print(f"[OK] saved: {out_csv}")

    return stats, summary

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz_x_y(npz_path: str, length: int):
    """
    读取 npz：
    - x: [N,L] 或 [N,1,L]
    - y: 可选 [N,2]（真实数据可能没有 y）
    """
    obj = np.load(npz_path, allow_pickle=True)
    if "x" not in obj:
        raise ValueError("npz 必须包含 key='x'")

    x = obj["x"]

    # x -> [N,1,L]
    if x.ndim == 2:
        if x.shape[1] != length:
            raise ValueError(f"x shape mismatch: expect [N,{length}], got {x.shape}")
        x = x[:, None, :]
    elif x.ndim == 3:
        if x.shape[1] != 1 or x.shape[2] != length:
            raise ValueError(f"x shape mismatch: expect [N,1,{length}], got {x.shape}")
    else:
        raise ValueError(f"x ndim must be 2 or 3, got {x.ndim}")

    # y 可能不存在
    y = None
    if "y" in obj:
        y = np.asarray(obj["y"])
        if y.ndim == 2:
            if y.shape[1] != 2:
                raise ValueError(f"y must be [N,2], got {y.shape}")
        else:
            y = y.reshape(y.shape[0], -1)
            if y.shape[1] != 2:
                raise ValueError(f"y must be [N,2], got {y.shape}")

        if y.shape[0] != x.shape[0]:
            raise ValueError(f"N mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} samples")

        y = torch.tensor(y, dtype=torch.float32)

    x = torch.tensor(x, dtype=torch.float32)
    return x, y


def minmax_per_sample_T(x: torch.Tensor, eps: float = 1e-32) -> torch.Tensor:
    """
    x: [N,1,T]
    每个样本沿 T 维做 min-max -> [0,1]
    eps 仅用于避免分母=0（常数序列）
    """
    x64 = x.to(torch.float64)
    x_min = x64.amin(dim=-1, keepdim=True)   # [N,1,1]
    x_max = x64.amax(dim=-1, keepdim=True)   # [N,1,1]
    denom = (x_max - x_min).clamp_min(eps)
    x_norm = (x64 - x_min) / denom
    return x_norm.to(dtype=x.dtype)


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    只做前向推理，返回：
    - y_pred_norm: [N,2]（注意：这是模型输出空间，可能是标准化/归一化后的 y）
    - y_true_raw:  [N,2] 或 None（npz 中的原始 y）
    """
    model.eval()
    preds = []
    trues = []

    for batch in tqdm(loader, desc="Infer", ncols=110):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch[0]
            y = None

        x = x.to(device)
        y_pred = model(x)  # [B,2]
        preds.append(y_pred.detach().cpu())

        if y is not None:
            trues.append(y.detach().cpu())

    y_pred_norm = torch.cat(preds, dim=0) if len(preds) > 0 else torch.empty(0, 2)
    y_true_raw = torch.cat(trues, dim=0) if len(trues) > 0 else None
    return y_pred_norm, y_true_raw


def inverse_y_to_real(y_pred_norm: torch.Tensor, ckpt: dict):
    """
    将模型输出的 y_pred_norm 反变换到“真实尺度”：
    - y_mode == 'none'：直接认为输出就是实尺度（仅考虑 y_scale）
    - y_mode == 'standardize'：*y_std + y_mean，再 / y_scale
    - y_mode == 'minmax'：*(y_max-y_min)+y_min，再 / y_scale
    """
    y_mode = ckpt.get("y_mode", "none")
    y_scale = float(ckpt.get("y_scale", 1.0))

    y_pred = y_pred_norm.clone()
    '''
    if y_mode == "standardize":
        if "y_mean" not in ckpt or "y_std" not in ckpt:
            raise ValueError("checkpoint 缺少 y_mean / y_std，无法反标准化")
        y_mean = torch.tensor(ckpt["y_mean"], dtype=torch.float32).view(1, 2)
        y_std = torch.tensor(ckpt["y_std"], dtype=torch.float32).view(1, 2)
        y_pred = y_pred * y_std + y_mean

    elif y_mode == "minmax":
        if "y_min" not in ckpt or "y_max" not in ckpt:
            raise ValueError("checkpoint 缺少 y_min / y_max，无法反归一化")
        y_min = torch.tensor(ckpt["y_min"], dtype=torch.float32).view(1, 2)
        y_max = torch.tensor(ckpt["y_max"], dtype=torch.float32).view(1, 2)
        y_pred = y_pred * (y_max - y_min) + y_min

    elif y_mode == "none":
        pass
    else:
        raise ValueError(f"未知 y_mode: {y_mode}")
    '''
    #train 里面没保存y_mode,这里直接用standardize
    if "y_mean" not in ckpt or "y_std" not in ckpt:
        raise ValueError("checkpoint 缺少 y_mean / y_std，无法反标准化")
    y_mean = torch.tensor(ckpt["y_mean"], dtype=torch.float32).view(1, 2)
    y_std = torch.tensor(ckpt["y_std"], dtype=torch.float32).view(1, 2)
    y_pred = y_pred * y_std + y_mean

    # 注意：训练时可能对 y 做过 y_scale 放大，这里要除回真实尺度
    if y_scale != 1.0:
        y_pred = y_pred / y_scale

    return y_pred


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    在真实尺度下计算：
    - 总体 mse/mae/rmse（两维一起平均）
    - 分别的 alpha/beta mse/mae/rmse
    """
    diff = y_pred - y_true  # [N,2]

    mse_a = (diff[:, 0] ** 2).mean().item()
    mse_b = (diff[:, 1] ** 2).mean().item()
    mae_a = diff[:, 0].abs().mean().item()
    mae_b = diff[:, 1].abs().mean().item()
    rmse_a = float(np.sqrt(mse_a))
    rmse_b = float(np.sqrt(mse_b))

    mse_all = (diff ** 2).mean().item()
    mae_all = diff.abs().mean().item()
    rmse_all = float(np.sqrt(mse_all))

    return {
        "mse_all": mse_all, "mae_all": mae_all, "rmse_all": rmse_all,
        "mse_alpha": mse_a, "mae_alpha": mae_a, "rmse_alpha": rmse_a,
        "mse_beta": mse_b, "mae_beta": mae_b, "rmse_beta": rmse_b,
    }


def export_csv_same_format(
    csv_path: str,
    y_true: Optional[torch.Tensor],
    y_pred: torch.Tensor,
    y_pred_phy: Optional[torch.Tensor] = None,
):
    """
    CSV 输出格式：按 resnet_main_0130.py 的 export_y_true_pred_csv 逻辑保持一致
    columns:
      sample_idx,
      Delta_alpha_true, Beta_true,
      Delta_alpha_pred, Beta_pred,
      Delta_alpha_diff, Beta_diff,
      Delta_alpha_diff_l1, Beta_diff_l1
    若 y_true 为 None：true/diff 相关列填 nan，但列名不变
    """
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    has_true = y_true is not None
    if has_true:
        y_true_np = y_true.cpu().numpy()
    else:
        y_true_np = None
    y_pred_np = y_pred.cpu().numpy()
    y_pred_phy_np = y_pred_phy.cpu().numpy() if y_pred_phy is not None else None

    header = [
        "sample_idx",
        "Delta_alpha_true", "Beta_true",
        "Delta_alpha_pred", "Beta_pred",
        "Delta_alpha_diff", "Beta_diff",
        "Delta_alpha_diff_l1", "Beta_diff_l1",
    ]

    # 追加传统方法（同一张表格里对齐输出）
    if y_pred_phy_np is not None:
        header += [
            "Delta_alpha_pred_phy", "Beta_pred_phy",
            "Delta_alpha_diff_phy", "Beta_diff_phy",
            "Delta_alpha_diff_l1_phy", "Beta_diff_l1_phy",
        ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        n = y_pred_np.shape[0]
        for i in range(n):
            pred_a = float(y_pred_np[i, 0])
            pred_b = float(y_pred_np[i, 1])

            if has_true:
                true_a = float(y_true_np[i, 0])
                true_b = float(y_true_np[i, 1])
                diff_a = pred_a - true_a
                diff_b = pred_b - true_b
                diff_a_l1 = abs(diff_a)
                diff_b_l1 = abs(diff_b)
            else:
                true_a = float("nan")
                true_b = float("nan")
                diff_a = float("nan")
                diff_b = float("nan")
                diff_a_l1 = float("nan")
                diff_b_l1 = float("nan")

            row = [
                i,
                true_a, true_b,
                pred_a, pred_b,
                diff_a, diff_b,
                diff_a_l1, diff_b_l1,
            ]

            if y_pred_phy_np is not None:
                phy_a = float(y_pred_phy_np[i, 0])
                phy_b = float(y_pred_phy_np[i, 1])

                if has_true and np.isfinite(phy_a) and np.isfinite(phy_b):
                    phy_diff_a = phy_a - true_a
                    phy_diff_b = phy_b - true_b
                    phy_diff_a_l1 = abs(phy_diff_a)
                    phy_diff_b_l1 = abs(phy_diff_b)
                else:
                    phy_diff_a = float("nan")
                    phy_diff_b = float("nan")
                    phy_diff_a_l1 = float("nan")
                    phy_diff_b_l1 = float("nan")

                row += [
                    phy_a, phy_b,
                    phy_diff_a, phy_diff_b,
                    phy_diff_a_l1, phy_diff_b_l1,
                ]

            w.writerow(row)


def main():
    # =======================
    # 你只需要改这几个路径/参数
    # =======================
    set_seed(42)

    ckpt_path = r"./output/ckpt_resnet1d50_reg2_0205_1/best.pt"   # 改成你的 best.pt / epoch_xxx.pt
    val_npz  = r"./npz_out/20260210/full.npz"                    # 改成你的 val/val/真实数据 npz（真实数据可无 y）,注意0210的full完全用作验证
    out_dir   = r"./val_out/20260227/"                            # 输出目录

    # 读取 checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    length = int(ckpt.get("length", 89))
    norm_x = bool(ckpt.get("norm_x", True))
    epoch  = int(ckpt.get("epoch", 0))
    y_mode = ckpt.get("y_mode", "none")
    y_scale = float(ckpt.get("y_scale", 1.0))

    # 读取数据
    x, y_true_raw = load_npz_x_y(val_npz, length=length)

    # 传统方法需要用“原始 x”（不能用 min-max 后的 x），否则幅值关系被破坏
    x_raw_for_phy = x.clone()

    # 与训练一致：x 每样本沿 T 做 min-max
    if norm_x:
        x = minmax_per_sample_T(x)

    # DataLoader（若无 y，也能跑）
    if y_true_raw is None:
        loader = DataLoader(TensorDataset(x), batch_size=512, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"), drop_last=False)
    else:
        loader = DataLoader(TensorDataset(x, y_true_raw), batch_size=512, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"), drop_last=False)

    # 构建模型（与训练脚本一致：resnet50_1d, num_classes=2）
    model = resnet50_1d(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 推理：得到模型输出空间的 y_pred_norm
    y_pred_norm, _ = infer(model, loader, device)

    # 反变换到真实尺度
    y_pred_real = inverse_y_to_real(y_pred_norm, ckpt)

    # ===============================
    # 传统方法（physics）反解：直接吃 x 和 lambda_nm
    # ===============================
    with np.load(val_npz, allow_pickle=True) as _npz:
        if "lambda_nm" not in _npz:
            raise KeyError("val.npz 中找不到 'lambda_nm'，传统方法需要波长轴。")
        lambda_nm = _npz["lambda_nm"]
        I0 = float(_npz["I0"]) if "I0" in _npz.files else 0.01  # 必须与你仿真一致

    y_pred_phy_np = solve_physics_batch(
        x_raw_for_phy.cpu().numpy(),
        lambda_nm,
        I0=I0,
    )
    y_pred_phy_real = torch.from_numpy(y_pred_phy_np)

    # y_true 使用 npz 原始 y（真实尺度）；注意训练时 y_scale 只是内部放大，不会改变 npz 的原值
    y_true_real = y_true_raw  # 可能为 None

    # 打印指标（若有 y）
    if y_true_real is not None:
        metrics = compute_metrics(y_true_real, y_pred_real)
        print(f"[ckpt] epoch={epoch}  y_mode={y_mode}  y_scale={y_scale}  norm_x={norm_x}  length={length}")
        print(f"[ALL]  mse={metrics['mse_all']:.6e}  mae={metrics['mae_all']:.6e}  rmse={metrics['rmse_all']:.6e}")
        print(f"[alpha] mse={metrics['mse_alpha']:.6e} mae={metrics['mae_alpha']:.6e} rmse={metrics['rmse_alpha']:.6e}")
        print(f"[beta]  mse={metrics['mse_beta']:.6e}  mae={metrics['mae_beta']:.6e}  rmse={metrics['rmse_beta']:.6e}")

        # physics 指标：只在“有解”的样本上统计
        mask_phy_np = np.isfinite(y_pred_phy_np).all(axis=1)
        mask_phy = torch.from_numpy(mask_phy_np)
        valid_ratio = float(mask_phy_np.mean()) * 100.0
        if mask_phy.any():
            metrics_phy = compute_metrics(y_true_real[mask_phy], y_pred_phy_real[mask_phy])
            print(f"[PHY]  valid={valid_ratio:.2f}%")
            print(f"[PHY]  mae_all={metrics_phy['mae_all']:.6e}  mae_alpha={metrics_phy['mae_alpha']:.6e}  mae_beta={metrics_phy['mae_beta']:.6e}")
        else:
            print(f"[PHY]  valid={valid_ratio:.2f}%  (no valid solutions)")
    else:
        print(f"[ckpt] epoch={epoch}  y_mode={y_mode}  y_scale={y_scale}  norm_x={norm_x}  length={length}")
        print("[warn] 当前 npz 无 y，无法计算误差指标，将只导出 y_pred 到 CSV（y_true/diff 填 nan）。")

    # CSV 路径：保持训练保存风格
    os.makedirs(out_dir, exist_ok=True)
    csv_name = f"best_epoch_{epoch:03d}_y_true_y_pred.csv"
    csv_path = os.path.join(out_dir, csv_name)

    export_csv_same_format(csv_path, y_true_real, y_pred_real, y_pred_phy_real)
    print(f"[OK] CSV saved: {csv_path}")

    if y_true_real is not None:
        stats, summary = group_std_by_exact_ytrue(
            csv_path=csv_path,
            out_csv=os.path.join(out_dir, "groupwise_std.csv"),
        )


if __name__ == "__main__":
    main()
