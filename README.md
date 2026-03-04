# ResNet1D Weak Measurement 工作流（MATLAB 仿真 → NPZ → 训练 → 验证）

本仓库提供一套端到端流程：  
**MATLAB 仿真生成数据（.mat） → Python 转换为训练数据（.npz） → 1D-ResNet 回归 (Delta_alpha, beta) → 推理验证与导出 CSV**。

---

## 工作流总览

1. **仿真生成 `.mat`**  
   `generate_T_PSD_3000x100_modeA_fixedI0.m`

2. **（可选）物理反解 baseline / 校正**  
   `physics_init_plus_ML_correction.m`

3. **`.mat` → `.npz` + 按组划分 train/test（避免泄露）**  
   `mat_to_npz_group_split.py`

4. **训练（train + 每轮 test 评估 + 保存 best.pt）**  
   `resnet_main_0130.py` + `resnet1d.py`

5. **验证/推理（加载 best.pt 导出 y_true/y_pred CSV）**  
   `resnet_test_main_0209.py`

---

## 文件说明

| 文件 | 语言 | 作用 | 主要输入 | 主要输出 |
|---|---|---|---|---|
| `generate_T_PSD_3000x100_modeA_fixedI0.m` | MATLAB | 仿真生成带噪序列并保存 | 脚本内参数 | `*.mat` |
| `physics_init_plus_ML_correction.m` | MATLAB | 物理反解 / baseline（可选） | `*.mat` | 统计表 / Excel（视脚本而定） |
| `mat_to_npz_group_split.py` | Python | 读取 `.mat`，生成 `train/test.npz` | `*.mat` | `full.npz / train.npz / test.npz` |
| `resnet1d.py` | Python | 1D-ResNet 网络定义 | `[B,1,L]` | `[B,2]` |
| `resnet_main_0130.py` | Python | 训练入口（保存 best.pt + 曲线 + CSV） | `train.npz/test.npz` | `best.pt` / `loss_history.csv` / 曲线图 / 预测 CSV |
| `resnet_test_main_0209.py` | Python | 推理验证入口（导出 CSV） | `best.pt` + `test/full.npz` | 指标 + 预测 CSV |

---

## 数据格式约定（关键）

### MATLAB `.mat`
- `Traw`：形状通常为 `[L, Nmeas, Ng]`（例如 `89 × 100 × N`）
- 标签：`Delta_alpha`、`beta`（长度为 `Ng`）
- 波长：`lambda`（长度为 `L`）

### Python `.npz`
- `x`：`[N, L]` 或 `[N, 1, L]`（训练脚本会统一成 `[N,1,L]`）
- `y`：`[N, 2]`，顺序固定为 `[Delta_alpha, beta]`

---

## 快速开始

### 1) MATLAB 生成 `.mat`
在 MATLAB 中运行：
- `generate_T_PSD_3000x100_modeA_fixedI0.m`  
（确保脚本里的输出目录存在/可写）

（可选）物理反解 baseline：
- `physics_init_plus_ML_correction.m`

---

### 2) 转换 `.mat` → `.npz`
在终端运行：

```powershell
python mat_to_npz_group_split.py
```

运行完成后应看到类似输出：
- `npz_out/YYYYMMDD/full.npz`
- `npz_out/YYYYMMDD/train.npz`
- `npz_out/YYYYMMDD/test.npz`

---

### 3) 训练
在 `resnet_main_0130.py` 的 `main()` 里修改路径（示例）：

```python
train_npz = r"./npz_out/20260206/train.npz"
test_npz  = r"./npz_out/20260206/test.npz"
length    = 89
save_dir  = r"./output/ckpt_resnet1d_xxx"
```

在终端运行：

```powershell
python resnet_main_0130.py
```

---

### 4) 验证 / 推理
在 `resnet_test_main_0209.py` 的 `main()` 里修改路径（示例）：

```python
ckpt_path = r"./output/ckpt_resnet1d_xxx/best.pt"
test_npz  = r"./npz_out/20260206/test.npz"
out_dir   = r"./test_out/20260206/"
```

在终端运行：

```powershell
python resnet_test_main_0209.py
```
