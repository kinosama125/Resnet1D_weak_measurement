# -*- coding: utf-8 -*-
"""
mat_to_npz_group_split.py
-----------------------------------------------------------------------------
用途
- 将 MATLAB v7.3（HDF5）格式的仿真 .mat 数据转换为 NPZ（x, y, lambda_nm），并按“组”划分
  train/val，避免数据泄露（同一组真实参数的重复测量不会同时出现在 train 与 val）。

输入（MATLAB v7.3 .mat）
- 期望字段（支持别名自动适配）：
  - Traw: 3D 或 2D（推荐 3D，包含重复测量）
  - Delta_alpha: [Ng]
  - beta: [Ng]
  - lambda / lambda_nm: [L]
- 重要注意（维度顺序）
  - 本脚本当前假设：Traw.shape == [Ng, repeats, L]
  - 但 MATLAB 常见保存方式可能是 [L, repeats, Ng]（例如 89×100×N）。
  - 若你发现转换后 Ng/L 对不上，请在读取 Traw 后做一次维度 permute（保持“Ng 在第 0 维、L 在最后一维”）。

输出（NPZ；写入 out_dir）
- full.npz : x [N, 1, L], y [N, 2], lambda_nm [L]
- train.npz/val.npz : 按组划分的子集（均包含 x, y, lambda_nm）
- 其中 y 的列顺序固定为：[Delta_alpha, beta]

主要接口
- convert_mat_to_npz_group_split(mat_path, out_dir, val_ratio=0.2, seed=42, mode="all")
  - mat_path   : 输入 .mat 路径
  - out_dir    : 输出目录（自动创建）
  - val_ratio : 测试组比例（按 group_ids 划分）
  - seed       : 划分随机种子（可复现）
  - mode:
      * "all"  : 每组 repeats 次重复都作为独立样本（N = Ng*repeats，推荐训练）
      * "mean" : 每组对 repeats 求均值后作为一个样本（N = Ng，更像传统反解输入）

实现要点
- 使用 h5py 读取 v7.3 .mat；字段名支持多候选（_read_dataset_any）
- 输出规整：
  - x: float32，最终统一成 [N, 1, L]
  - y: float32，[N, 2]
- 分组划分：
  - group_ids 记录每个样本所属参数组
  - train/val 的 group_ids 必须完全不相交（内置泄露检查）

依赖
- numpy
- h5py（读取 MATLAB v7.3）
-----------------------------------------------------------------------------
"""

import os
import numpy as np
import h5py


def _read_dataset_any(f: h5py.File, names):
    """按候选名字读取数据集（适配不同 mat 字段命名）"""
    for n in names:
        if n in f:
            return f[n][:]
    raise KeyError(f"在 .mat 里找不到任何一个字段：{names}. 现有字段={list(f.keys())}")


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _save_npz_full_train_val(
    x: np.ndarray,
    y: np.ndarray,
    lambda_nm: np.ndarray,
    group_ids: np.ndarray,
    out_dir: str,
    val_ratio: float,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 形状检查 ----
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    lambda_nm = np.asarray(lambda_nm, dtype=np.float32).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)

    if x.ndim == 2:
        x = x[:, None, :]  # [N,1,L]
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"x 形状必须是 [N,1,L] 或 [N,L]，当前 {x.shape}")
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y 形状必须是 [N,2]，当前 {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y 样本数不一致：{x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] != group_ids.shape[0]:
        raise ValueError(f"group_ids 长度不一致：{group_ids.shape[0]} vs N={x.shape[0]}")
    if x.shape[2] != lambda_nm.shape[0]:
        raise ValueError(f"L 不一致：x 的 L={x.shape[2]} vs lambda_nm 的 L={lambda_nm.shape[0]}")

    # ---- 保存 full ----
    full_path = os.path.join(out_dir, "full.npz")
    np.savez_compressed(full_path, x=x, y=y, lambda_nm=lambda_nm)
    print(f"[OK] Saved: {full_path}")
    print(f"     x: {x.shape} float32, y: {y.shape} float32, lambda_nm: {lambda_nm.shape} float32")

    # ---- 按组划分 train/val（避免泄露）----
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(group_ids)
    rng.shuffle(uniq_groups)

    n_val_groups = int(round(len(uniq_groups) * val_ratio))
    n_val_groups = max(1, min(len(uniq_groups) - 1, n_val_groups))  # 至少 1 组测试，至少 1 组训练

    val_groups = uniq_groups[:n_val_groups]
    is_val = np.isin(group_ids, val_groups)
    is_train = ~is_val

    train_path = os.path.join(out_dir, "train.npz")
    val_path = os.path.join(out_dir, "val.npz")
    np.savez_compressed(train_path, x=x[is_train], y=y[is_train], lambda_nm=lambda_nm)
    np.savez_compressed(val_path,  x=x[is_val],  y=y[is_val],  lambda_nm=lambda_nm)

    # ---- 泄露检查：train/val 组必须不相交 ----
    train_groups = set(np.unique(group_ids[is_train]).tolist())
    val_groups_set = set(np.unique(group_ids[is_val]).tolist())
    if not train_groups.isdisjoint(val_groups_set):
        raise RuntimeError("Data leakage detected: some groups appear in both train and val!")

    print(f"[OK] Saved: {train_path}  (samples={int(is_train.sum())}, groups={len(train_groups)})")
    print(f"[OK] Saved: {val_path}   (samples={int(is_val.sum())}, groups={len(val_groups_set)})")


def convert_mat_to_npz_group_split(
    mat_path: str,
    out_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    mode: str = "all",
):
    """
    mat_path: 你的 MATLAB v7.3 .mat 路径
    mode:
      - "all" : 每组 repeats 次重复都当作独立样本（推荐用于训练，N=Ng*repeats）
      - "mean": 每组先对 repeats 取均值（更像传统反解用的 Tmean，N=Ng）
    """
    if mode not in ("all", "mean"):
        raise ValueError("mode 只能是 'all' 或 'mean'")

    with h5py.File(mat_path, "r") as f:
        # 你这个文件的字段：Traw (3000,100,89), Delta_alpha (1,3000), beta (1,3000), lambda (1,89)
        Traw = _read_dataset_any(f, ["Traw", "T", "X", "T_meas"])  # [Ng, repeats, L] 或 [Ng, L]
        da = _read_dataset_any(f, ["Delta_alpha", "delta_alpha", "DA", "da"])
        be = _read_dataset_any(f, ["beta", "b", "BETA"])
        lam = _read_dataset_any(f, ["lambda", "lambda_nm", "lam", "wavelength"])

    # ---- 规整形状 ----
    da = _ensure_1d(da).astype(np.float32)  # [Ng]
    be = _ensure_1d(be).astype(np.float32)  # [Ng]
    lambda_nm = _ensure_1d(lam).astype(np.float32)  # [L]

    if da.shape[0] != be.shape[0]:
        raise ValueError(f"Delta_alpha 和 beta 长度不一致：{da.shape[0]} vs {be.shape[0]}")

    # Traw 可能是 [Ng, repeats, L]，也可能已是 [Ng, L]
    Traw = np.asarray(Traw)
    if Traw.ndim == 3:
        Ng, repeats, L = Traw.shape
        if da.shape[0] != Ng:
            raise ValueError(f"DA/beta 的 Ng={da.shape[0]} 与 Traw 的 Ng={Ng} 不一致")
        if lambda_nm.shape[0] != L:
            raise ValueError(f"lambda_nm 的 L={lambda_nm.shape[0]} 与 Traw 的 L={L} 不一致")

        y_group = np.stack([da, be], axis=1).astype(np.float32)  # [Ng,2]

        if mode == "all":
            # 展开为 N=Ng*repeats
            x = Traw.reshape(Ng * repeats, L).astype(np.float32)          # [N,L]
            y = np.repeat(y_group, repeats=repeats, axis=0).astype(np.float32)  # [N,2]
            group_ids = np.repeat(np.arange(Ng, dtype=np.int64), repeats) # [N]
        else:  # mean
            x = Traw.mean(axis=1).astype(np.float32)                     # [Ng,L]
            y = y_group
            group_ids = np.arange(Ng, dtype=np.int64)

    elif Traw.ndim == 2:
        Ng, L = Traw.shape
        if da.shape[0] != Ng:
            raise ValueError(f"DA/beta 的 Ng={da.shape[0]} 与 Traw 的 Ng={Ng} 不一致")
        if lambda_nm.shape[0] != L:
            raise ValueError(f"lambda_nm 的 L={lambda_nm.shape[0]} 与 Traw 的 L={L} 不一致")
        x = Traw.astype(np.float32)
        y = np.stack([da, be], axis=1).astype(np.float32)
        group_ids = np.arange(Ng, dtype=np.int64)

    else:
        raise ValueError(f"Traw 维度不支持：{Traw.shape}（期望 2D 或 3D）")

    _save_npz_full_train_val(
        x=x,
        y=y,
        lambda_nm=lambda_nm,
        group_ids=group_ids,
        out_dir=out_dir,
        val_ratio=val_ratio,
        seed=seed,
    )


def main():
    # ===================== 在这里改参数（不走命令行） =====================
    mat_path = r"D:\Wang\07_ResNet1D\ResNet1D_weak_measurement\data\20260302\Traw_3000x100_PSD_ModeA_fixedI0_withNoise.mat"
    out_dir = r"D:\Wang\07_ResNet1D\ResNet1D_weak_measurement\data\20260302\npzout"

    val_ratio = 0.3
    seed = 42

    # "all": 输出 N=Ng*repeats（例如 3000*100=300000）
    # "mean": 输出 N=Ng（例如 3000）
    mode = "all"

    convert_mat_to_npz_group_split(
        mat_path=mat_path,
        out_dir=out_dir,
        val_ratio=val_ratio,
        seed=seed,
        mode=mode,
    )


if __name__ == "__main__":
    main()
