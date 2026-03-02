"""
resnet_main_0130.py
-----------------------------------------------------------------------------
用途
- 训练 1D-ResNet 回归模型：输入为单条 1D 序列 x（光谱/PSD），输出为两参数 y=[Delta_alpha, beta]。
- 数据来自 mat_to_npz_group_split.py 生成的 train.npz / test.npz（键：x, y；可额外带 lambda_nm 但训练不依赖它）。

输入数据（NPZ）
- x: [N, L] 或 [N, 1, L]，脚本内统一为 torch.float32 的 [N, 1, L]
- y: [N, 2]，列顺序为 [Delta_alpha, beta]
- 关键超参：length(L), epochs, batch_size, lr, weight_decay, seed, save_dir 等在 main() 中配置。

预处理与标签空间
- x 预处理（可开关 norm_x）：
  - 每个样本沿 T/L 维做 min-max 归一化到 [0,1]（minmax_per_sample_T）
- y 处理：
  - 仅用训练集统计量做 z-score 标准化：y_norm = (y - y_mean)/y_std
  - 训练反传在 y_norm 空间；评估指标/日志统一在“真实尺度 y_raw”上计算（通过反标准化实现）。

模型与损失
- 模型：resnet50_1d(in_channels=1, num_classes=2)
  - 注意：文件名虽包含 “18”，但当前脚本实际实例化的是 resnet50_1d（以代码为准）
- 损失（训练空间 y_norm）：
  - loss = 0.01 * MSE(alpha) + 1.0 * MSE(beta)
- 优化器：AdamW
- 学习率调度：StepLR(step_size=5, gamma=0.1) —— 每 5 个 epoch 将 lr 乘以 0.1
- grad_clip 参数保留，但剪裁代码目前被注释（不生效）。

主要函数（功能/输入输出）
- load_npz_dataset(npz_path, length) -> (x:Tensor[N,1,L], y:Tensor[N,2])
- minmax_per_sample_T(x[N,1,L]) -> x_norm[N,1,L]
- standardize_y_train_stats(y_train_raw[N,2]) -> (y_mean[1,2], y_std[1,2])
- evaluate(model, loader, ..., y_mean, y_std) -> (loss_norm, mse, mae, rmse, mse_a, mse_b, mae_a, mae_b)
- dump_y_and_pred_csv(...) -> 保存 “真实尺度” y_true/y_pred/diff 到 CSV
- train(...) -> 完整训练流程（每 epoch eval，按 test_mse_real 选 best）
- main() -> 配置路径与超参，调用 train()

输出（写入 save_dir）
- best.pt : 保存最佳模型与关键统计量
  - 包含：epoch, model_state, optimizer_state, best_mse_real, length, norm_x, y_mean, y_std
- best_epoch_XXX_y_true_y_pred.csv : 对 test 集导出真实尺度预测
  - 列：Delta_alpha_true/Beta_true/Delta_alpha_pred/Beta_pred/差值与 L1 等
- loss_curve*.png, loss_history.csv : 训练/测试曲线与每 epoch 记录

下游衔接
- resnet_test_main_0209.py 会加载 best.pt 并在任意 npz 上推理/导出 CSV，并可对齐传统反解结果。
-----------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from resnet1d import resnet18_1d, resnet34_1d, resnet50_1d

import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """固定随机种子，保证可复现（注意：会牺牲一点 cudnn 性能）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz_dataset(npz_path: str, length: int = 45):
    """
    读取你保存的 npz（键：x,y）
    - x: 期望是 [N,L] 或 [N,1,L]，最终统一成 torch.float32 的 [N,1,L]
    - y: 期望是 [N,2]（两参数回归）
    """
    obj = np.load(npz_path, allow_pickle=True)
    x = obj["x"]
    y = obj["y"]

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

    # y -> [N,2]
    y = np.asarray(y)
    if y.ndim == 1:
        raise ValueError("y must be [N,2] for 2-target regression, but got 1D array.")
    if y.ndim == 2:
        if y.shape[1] != 2:
            raise ValueError(f"y must be [N,2], got {y.shape}")
    else:
        y = y.reshape(y.shape[0], -1)
        if y.shape[1] != 2:
            raise ValueError(f"y must be [N,2], got {y.shape}")

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def minmax_per_sample_T(x: torch.Tensor, eps: float = 1e-32) -> torch.Tensor:
    """
    x: [N,1,T]
    对每个样本自己，在 T 维做 min-max -> [0,1]
    eps 只用于避免分母为0（常数序列）
    """
    # 用 float64 做 min/max 更稳（不改变你整体 dtype 习惯）
    x64 = x.to(torch.float64)
    x_min = x64.amin(dim=-1, keepdim=True)   # [N,1,1]
    x_max = x64.amax(dim=-1, keepdim=True)   # [N,1,1]
    denom = (x_max - x_min).clamp_min(eps)
    x_norm = (x64 - x_min) / denom
    return x_norm.to(dtype=x.dtype)


def standardize_y_train_stats(y_train_raw: torch.Tensor, eps: float = 1e-12):
    """
    y_train_raw: [N,2]
    仅用训练集统计量做 z-score（解决 y 的量级/方差太小导致“训练后期不动”问题）
    y_norm = (y - mean) / std
    """
    mean = y_train_raw.mean(dim=0, keepdim=True)                 # [1,2]
    std = y_train_raw.std(dim=0, keepdim=True).clamp_min(eps)    # [1,2]
    return mean, std


def plot_losses(train_losses, test_losses, save_path=None):
    """画总 MSE（真实尺度）随 epoch 的曲线"""
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="train mse (orig)")
    plt.plot(epochs, test_losses, label="test mse (orig)")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("Train/Test MSE (Original Scale)")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             criterion: nn.Module,
             y_mean: torch.Tensor,
             y_std: torch.Tensor):
    """
    loader 返回: x, y_norm, y_raw
    评估要点：
    1) pred_norm = model(x)  —— 模型输出在“标准化空间”
    2) pred_raw  = pred_norm * y_std + y_mean —— 反标准化回“真实尺度”
    3) 指标（MSE/MAE/RMSE）一律在真实尺度(y_raw)上计算
    4) 同时把 α(第0维) 与 β(第1维) 的误差分开统计并返回
    """
    model.eval()

    total_loss_norm = 0.0   # 标准化空间的 loss（仅参考）
    total = 0

    # 总体（两维合并）
    se_sum = 0.0
    ae_sum = 0.0
    denom = 0

    # 分维度：α 和 β
    se_a = 0.0
    se_b = 0.0
    ae_a = 0.0
    ae_b = 0.0
    denom_a = 0
    denom_b = 0

    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    for x, y_norm, y_raw in loader:
        x = x.to(device)
        y_norm = y_norm.to(device)
        y_raw = y_raw.to(device)

        pred_norm = model(x)                         # [B,2]
        loss_norm = criterion(pred_norm, y_norm)     # 标准化空间的 loss

        bs = x.size(0)
        total_loss_norm += loss_norm.item() * bs
        total += bs

        # 反标准化到真实尺度
        pred_raw = pred_norm * y_std + y_mean        # [B,2]
        diff = pred_raw - y_raw                      # [B,2]

        # 总体（两维一起）
        se_sum += (diff * diff).sum().item()
        ae_sum += diff.abs().sum().item()
        denom += diff.numel()

        # 分维度（α/β）
        se_a += (diff[:, 0] * diff[:, 0]).sum().item()
        se_b += (diff[:, 1] * diff[:, 1]).sum().item()
        ae_a += diff[:, 0].abs().sum().item()
        ae_b += diff[:, 1].abs().sum().item()
        denom_a += diff[:, 0].numel()
        denom_b += diff[:, 1].numel()

    avg_loss_norm = total_loss_norm / max(total, 1)

    mse = se_sum / max(denom, 1)
    mae = ae_sum / max(denom, 1)
    rmse = float(np.sqrt(mse))

    mse_a = se_a / max(denom_a, 1)
    mse_b = se_b / max(denom_b, 1)
    mae_a = ae_a / max(denom_a, 1)
    mae_b = ae_b / max(denom_b, 1)

    return avg_loss_norm, mse, mae, rmse, mse_a, mse_b, mae_a, mae_b


@torch.no_grad()
def dump_y_and_pred_csv(model: nn.Module,
                        loader: DataLoader,
                        device: torch.device,
                        y_mean: torch.Tensor,
                        y_std: torch.Tensor,
                        save_csv_path: str):
    """
    你要求的：存储 pth 的同时，把输入 y（真实尺度 y_raw）和 pred y（真实尺度 pred_raw）导出 CSV
    - CSV 每行对应一个样本
    - 字段包含：y_true / y_pred / diff
    """
    model.eval()
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    ys = []
    ps = []

    for x, _, y_raw in loader:
        x = x.to(device)
        pred_norm = model(x)                   # [B,2] 标准化空间
        pred_raw = pred_norm * y_std + y_mean  # [B,2] 真实尺度

        ys.append(y_raw.numpy())
        ps.append(pred_raw.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)   # [N,2]
    y_pred = np.concatenate(ps, axis=0)   # [N,2]

    df = pd.DataFrame({
        "Delta_alpha_true": y_true[:, 0],
        "Beta_true":        y_true[:, 1],
        "Delta_alpha_pred": y_pred[:, 0],
        "Beta_pred":        y_pred[:, 1],
        "Delta_alpha_diff": (y_pred[:, 0] - y_true[:, 0]),
        "Beta_diff": (y_pred[:, 1] - y_true[:, 1]),
        "Delta_alpha_diff_l1": np.abs(y_pred[:, 0] - y_true[:, 0]),
        "Beta_diff_l1":        np.abs(y_pred[:, 1] - y_true[:, 1]),
    })
    df.to_csv(save_csv_path, index=False)


def train(
    train_npz: str,
    test_npz: str,
    length: int = 45,
    epochs: int = 50,
    batch_size: int = 128,
    test_batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    seed: int = 42,
    save_dir: str = "./ckpt_resnet1d18_reg2",
    norm_x: bool = True,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data ----------
    x_train, y_train_raw = load_npz_dataset(train_npz, length=length)
    x_test,  y_test_raw  = load_npz_dataset(test_npz,  length=length)

    # 只归一化输入 x：每个样本沿 T -> 0~1（你原逻辑不改）
    if norm_x:
        x_train = minmax_per_sample_T(x_train)
        x_test  = minmax_per_sample_T(x_test)

    # y：训练用 z-score（仅用 train 统计量），但日志/曲线/评估一律用真实尺度
    y_mean, y_std = standardize_y_train_stats(y_train_raw, eps=1e-12)  # [1,2]
    y_train = (y_train_raw - y_mean) / y_std
    y_test  = (y_test_raw  - y_mean) / y_std

    # DataLoader：把 y_norm 和 y_raw 都带上，便于训练/评估分别使用
    train_loader = DataLoader(
        TensorDataset(x_train, y_train, y_train_raw),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test, y_test_raw),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---------- model ----------
    model = resnet50_1d(in_channels=1, num_classes=2).to(device)

    # shape check（不改）
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(4, 1, length).to(device)
        out = model(dummy)
    print(f"[shape check] in={tuple(dummy.shape)} out={tuple(out.shape)} (expect [B,2])")

    # ---------- loss / optim ----------
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 你要求的优化点 1：scheduler（每个 epoch 学习率衰减 0.98）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_mse = float("inf")

    # 你原来的总曲线：真实尺度 MSE（总）
    train_losses = []
    test_losses = []

    # 你要求的优化点 2：α/β 损失分别存储（真实尺度）
    train_losses_a = []
    train_losses_b = []
    test_losses_a = []
    test_losses_b = []

    # 同步保存每轮学习率，方便你回看 scheduler 是否生效
    lr_history = []

    os.makedirs(save_dir, exist_ok=True)

    # 预先把 y_mean/y_std 放到 device，避免每个 iter 反复搬运
    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)

    # ---------- train loop ----------
    for epoch in range(1, epochs + 1):
        model.train()

        # 训练集真实尺度统计：总/分维度
        se_sum_epoch = 0.0
        denom_epoch = 0

        se_a_epoch = 0.0
        se_b_epoch = 0.0
        denom_a_epoch = 0
        denom_b_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=110)
        for x, y_norm, y_raw in pbar:
            x = x.to(device)
            y_norm = y_norm.to(device)
            y_raw = y_raw.to(device)

            optimizer.zero_grad(set_to_none=True)

            # ---------- forward ----------
            pred_norm = model(x)                 # [B,2]（标准化空间输出）

            # 你要求的优化点 2：α/β 的 loss 分开算（训练空间）
            loss_a_norm = criterion(pred_norm[:, 0], y_norm[:, 0])
            loss_b_norm = criterion(pred_norm[:, 1], y_norm[:, 1])
            loss_norm = 0.01 * loss_a_norm + loss_b_norm

            # ---------- backward ----------
            loss_norm.backward()

            # 你原来是注释掉的，我保持不动
            '''
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            '''
            optimizer.step()

            # ---------- 真实尺度的统计（只用于日志/曲线，不参与反传）----------
            pred_raw = pred_norm * y_std_dev + y_mean_dev
            diff = pred_raw - y_raw

            # 总体
            se_sum_epoch += (diff * diff).sum().item()
            denom_epoch += diff.numel()

            # 分维度（α/β）
            se_a_epoch += (diff[:, 0] * diff[:, 0]).sum().item()
            se_b_epoch += (diff[:, 1] * diff[:, 1]).sum().item()
            denom_a_epoch += diff[:, 0].numel()
            denom_b_epoch += diff[:, 1].numel()

            train_mse_real = se_sum_epoch / max(denom_epoch, 1)
            pbar.set_postfix(train_mse_real=f"{train_mse_real:.6g}")

        # ---------- test each epoch ----------
        _, mse, mae, rmse, mse_a, mse_b, mae_a, mae_b = evaluate(
            model, test_loader, device, criterion, y_mean=y_mean, y_std=y_std
        )

        # 训练集真实尺度 mse（总/α/β）
        train_mse_real = se_sum_epoch / max(denom_epoch, 1)
        train_mse_a = se_a_epoch / max(denom_a_epoch, 1)
        train_mse_b = se_b_epoch / max(denom_b_epoch, 1)

        # 记录曲线：全部真实尺度
        train_losses.append(train_mse_real)
        test_losses.append(mse)

        train_losses_a.append(train_mse_a)
        train_losses_b.append(train_mse_b)
        test_losses_a.append(mse_a)
        test_losses_b.append(mse_b)

        # 记录 lr（scheduler.step 之前的当前 lr）
        cur_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(cur_lr)

        msg = (
            f"[E{epoch}] "
            f"train_mse={train_mse_real:.6g} | test_mse={mse:.6g} test_rmse={rmse:.6g} | "
            f"a_mse(train/test)={train_mse_a:.6g}/{mse_a:.6g} | "
            f"b_mse(train/test)={train_mse_b:.6g}/{mse_b:.6g} | "
            f"a_mae={mae_a:.6g} b_mae={mae_b:.6g} | lr={cur_lr:.3e}"
        )
        if norm_x:
            msg += " (x minmax per-sample, y zscore train-stats)"
        print(msg)

        # ---------- save best ----------
        # 你原逻辑：按真实尺度 test_mse 选 best（不改，只在 best 时额外导出 CSV）
        if mse < best_mse:
            best_mse = mse
            ckpt_path = os.path.join(save_dir, "best.pt")

            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_mse_real": best_mse,
                "length": length,
                "norm_x": norm_x,
                # 推理/继续训练时必须用同一套 y_mean/y_std
                "y_mean": y_mean.cpu().numpy(),
                "y_std": y_std.cpu().numpy(),
            }
            torch.save(payload, ckpt_path)
            print(f"  -> saved best to {ckpt_path} (best_mse_real={best_mse:.6g})")

            # 你要求的优化点 3：保存 best.pt 的同时导出 y_true/y_pred CSV（真实尺度）
            csv_path = os.path.join(save_dir, f"best_epoch_{epoch:03d}_y_true_y_pred.csv")
            dump_y_and_pred_csv(model, test_loader, device, y_mean=y_mean, y_std=y_std, save_csv_path=csv_path)
            print(f"  -> saved csv to {csv_path}")

        # ---------- scheduler step ----------
        # 你要求：每 epoch 衰减 0.5
        scheduler.step()

    # 画总 loss 曲线（真实尺度）
    plot_losses(train_losses, test_losses, save_path=os.path.join(save_dir, "loss_curve.png"))
    plot_losses(train_losses_a, test_losses_a, save_path=os.path.join(save_dir, "loss_curve_a.png"))
    plot_losses(train_losses_b, test_losses_b, save_path=os.path.join(save_dir, "loss_curve_b.png"))

    # 同时把 α/β 分开的 loss 也落盘（CSV），便于你做分析/画图
    loss_csv = os.path.join(save_dir, "loss_history.csv")
    df_hist = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1, dtype=np.int32),
        "lr": np.array(lr_history, dtype=np.float64),

        "train_mse": np.array(train_losses, dtype=np.float64),
        "test_mse":  np.array(test_losses, dtype=np.float64),

        "train_mse_alpha": np.array(train_losses_a, dtype=np.float64),
        "train_mse_beta":  np.array(train_losses_b, dtype=np.float64),
        "test_mse_alpha":  np.array(test_losses_a, dtype=np.float64),
        "test_mse_beta":   np.array(test_losses_b, dtype=np.float64),
    })
    df_hist.to_csv(loss_csv, index=False)


def main():
    # =======================
    # 你在这里改参数就行
    # =======================
    train_npz = r"./npz_out/20260210/train.npz"
    test_npz  = r"./npz_out/20260210/test.npz"   # 你说目前先保持 test（后续再换 val/test）

    length = 89
    epochs = 50
    batch_size = 128
    test_batch_size = 512

    lr = 1e-3
    weight_decay = 1e-2
    grad_clip = 1.0

    seed = 42
    save_dir = r"./output/ckpt_resnet1d50_reg2_0302"
    norm_x = True

    train(
        train_npz=train_npz,
        test_npz=test_npz,
        length=length,
        epochs=epochs,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        seed=seed,
        save_dir=save_dir,
        norm_x=norm_x,
    )


if __name__ == "__main__":
    main()

