%% physics_init_plus_ML_correction.m
% -------------------------------------------------------------------------
% 用途
% - 对“Mode A（固定 I0）”生成的原始测量谱 Traw 做闭式反演（closed-form recovery），
%   对每一次测量估计 Delta_alpha_hat 与 beta_hat，并统计：
%   1) 全部测量的逐次反演结果
%   2) 每个参数组内（重复测量）的均值/标准差（反映组内波动）
%   3) 传统方法“无解/全 NaN 组”的失败率（fail_rate）
%
% 输入（脚本内配置路径）
% - inMat : 由 generate_T_PSD_3000x100_modeA_fixedI0.m 生成的 MAT 文件路径
%           需要包含变量：lambda, omega, Traw, Delta_alpha, beta, I0
% - outdir: 输出目录
%
% 主要输出（写入 outdir；注意：当前文件名写死了 “30x1000”，但实际由数据决定）
% 1) DA_beta_hat_30x1000_ModeA_closedform.xlsx（Sheet: all_measurements）
%    - 列：group_id, DA_true, beta_true, DA_hat, beta_hat
%    - 行数：Nmeas * Ng
% 2) std_over_30_measurements_ModeA_closedform.xlsx（Sheet: std_per_group）
%    - 列：group_id, DA_true, beta_true, DA_mean, DA_std, beta_mean, beta_std
%    - 行数：Ng
%
% 核心流程要点
% - 读取后会对 omega 升序排序，并同步重排 Traw（保持 domega 权重一致）
% - 对每个 (group g, measurement k)：
%   * 计算 M0 = Σ T(ω) Δω，M1 = Σ ω·T(ω) Δω
%   * 根据解析式求 sin(phi)，并对 cos(phi) 的两个分支分别尝试
%   * 将 phi -> Delta_alpha = 0.5*phi，并检查 Delta_alpha 范围合法性
%   * 由 acosh(...) 反解 beta
% - 统计 L1 误差与 all-NaN 组比例（fail_rate），用于解释传统方法稳定性
%
% 数据形状约定（MATLAB 维度）
% - Traw: [L, Nmeas, Ng]
% - DA_hat / b_hat: [Nmeas, Ng]（无解填 NaN）
%
% 下游衔接
% - 该脚本用于“传统方法基线”对比；（注意在resnet_val_main_0209.py已经用python实现，此文件仅作为验证）
% - Python 的 resnet_val_main_0209.py 也会调用对应的 Python 版 solve_physics_batch
%   在同一 CSV 内对齐输出网络预测与传统预测。
% -------------------------------------------------------------------------

clc; clear;

%% ===== 1) 读入原始数据 =====
outdir = 'F:\DL\Code\07_ResNet1D\data_generate_matlab\20260210\';
inMat = 'F:\DL\Code\07_ResNet1D\data_generate_matlab\20260210\Traw_3000x100_PSD_ModeA_fixedI0_withNoise.mat';
S = load(inMat);   % 需要：lambda, omega, Delta_alpha, beta, I0, Traw

lambda = S.lambda(:);
omega  = S.omega(:);
Traw   = S.Traw;          % 89 × Nmeas × Ngroup
DA_true_all = S.Delta_alpha;
b_true_all  = S.beta;
I0 = S.I0;

[nLam, Nmeas, Ng] = size(Traw);
fprintf('Loaded Traw: %d lambda × %d meas × %d groups\n', nLam, Nmeas, Ng);

%% ===== 2) 常数 =====
c   = 3e8;
tau = 2.18e-16;

omega0 = 2*pi*c/(623.61e-9);
delta  = 2*pi*c*(5e-9)/(623.61e-9)^2;

Aexp = exp(-(delta^2)*(tau^2));
K    = Aexp * (delta^2) * tau;

% omega 网格权重
[omega, idx] = sort(omega, 'ascend');
domega = abs(gradient(omega));

Traw = Traw(idx,:,:);

%% ===== 3) 逐次反演 =====
DA_hat = nan(Nmeas, Ng);
b_hat  = nan(Nmeas, Ng);

DeltaAlphaRange = [-0.003, 0.003];

for g = 1:Ng
    for k = 1:Nmeas
        T = double(Traw(:,k,g));

        % omega-矩（不归一化）
        M0 = sum(T .* domega);
        M1 = sum((omega .* T) .* domega);

        % ---- sin(phi) ----
        s = -2 * (M1 - omega0*M0) / (I0 * K);
        if ~isfinite(s), continue; end
        s = max(-1, min(1, s));

        % 分支：cos(phi)=±sqrt(1-s^2)
        cabs = sqrt(max(0, 1 - s^2));
        cands = [cabs, -cabs];

        solved = false;
        for kk = 1:2
            cphi = cands(kk);
            phi  = atan2(s, cphi);
            DA   = 0.5 * phi;

            if DA < DeltaAlphaRange(1) || DA > DeltaAlphaRange(2)
                continue;
            end

            C2b = (2*M0)/I0 + Aexp*cphi;
            if C2b < 1, continue; end

            b = 0.5 * acosh(C2b);

            DA_hat(k,g) = DA;
            b_hat(k,g)  = b;
            solved = true;
            break;
        end

        if ~solved
            DA_hat(k,g) = NaN;
            b_hat(k,g)  = NaN;
        end
    end
end

fprintf('Valid solutions: %.2f %%\n', ...
    100*nnz(isfinite(DA_hat(:)))/numel(DA_hat));

%% ===== 4) 每组 30 次统计 =====
% DA_mean = nan(Ng,1);  DA_std = nan(Ng,1);
% b_mean  = nan(Ng,1);  b_std  = nan(Ng,1);
% 
% for g = 1:Ng
%     DA_mean(g) = mean(DA_hat(:,g),'omitnan');
%     DA_std(g)  = std(DA_hat(:,g),'omitnan');
%     b_mean(g)  = mean(b_hat(:,g),'omitnan');
%     b_std(g)   = std(b_hat(:,g),'omitnan');
% end
% 
% DA_delta = DA_mean - DA_true_all;
% b_delta = b_mean - b_true_all;
% 
% DA_rmse = sqrt(mean(DA_delta.^2));
% b_rmse = sqrt(mean(b_delta.^2));
%% ===== 4) 每组 30 次统计 =====
DA_mean = nan(Ng,1);  DA_std = nan(Ng,1);
b_mean  = nan(Ng,1);  b_std  = nan(Ng,1);

DA_valid_cnt = zeros(Ng,1);
b_valid_cnt  = zeros(Ng,1);

for g = 1:Ng
    DA_valid_cnt(g) = sum(~isnan(DA_hat(:,g)));
    b_valid_cnt(g)  = sum(~isnan(b_hat(:,g)));

    if DA_valid_cnt(g) > 0
        DA_mean(g) = mean(DA_hat(:,g), 'omitnan');
        DA_std(g)  = std(DA_hat(:,g),  'omitnan');
    end

    if b_valid_cnt(g) > 0
        b_mean(g) = mean(b_hat(:,g), 'omitnan');
        b_std(g)  = std(b_hat(:,g),  'omitnan');
    end
end

%% 误差（逐组）
DA_delta = DA_mean - DA_true_all;
b_delta  = b_mean  - b_true_all;

%% ===== 只在“有有效解”的组上算 L1 =====
idx_DA = ~isnan(DA_delta);
idx_b  = ~isnan(b_delta);

if any(idx_DA)
    DA_l1 = mean(abs(DA_delta(idx_DA)));
else
    DA_l1 = NaN;  % 所有组都失败
end

if any(idx_b)
    b_l1 = mean(abs(b_delta(idx_b)));
else
    b_l1 = NaN;
end

%% ===== 额外输出：全 NaN 组的失败率（非常关键）=====
DA_fail_rate = mean(DA_valid_cnt == 0);
b_fail_rate  = mean(b_valid_cnt  == 0);

fprintf('[DA] L1 loss=%.3e, fail_rate=%.2f%% (all-NaN groups)\n', DA_l1, 100*DA_fail_rate);
fprintf('[b ] L1 loss=%.3e, fail_rate=%.2f%% (all-NaN groups)\n', b_l1,  100*b_fail_rate);


%% ===== 5) 导出逐次结果 =====
rows = Nmeas*Ng;
group_id = repelem((1:Ng).', Nmeas);
DA_true_rep = repelem(DA_true_all(:), Nmeas);
b_true_rep  = repelem(b_true_all(:),  Nmeas);

Tout = table(group_id, DA_true_rep, b_true_rep, ...
             DA_hat(:), b_hat(:), ...
    'VariableNames', {'group_id','DA_true','beta_true','DA_hat','beta_hat'});

writetable(Tout, strcat(outdir,'DA_beta_hat_30x1000_ModeA_closedform.xlsx'), 'Sheet','all_measurements');

%% ===== 6) 导出每组 STD =====
Tstd = table((1:Ng).', DA_true_all(:), b_true_all(:), ...
             DA_mean, DA_std, b_mean, b_std, ...
    'VariableNames', {'group_id','DA_true','beta_true', ...
                      'DA_mean','DA_std','beta_mean','beta_std'});

writetable(Tstd, strcat(outdir,'std_over_30_measurements_ModeA_closedform.xlsx'), ...
           'Sheet','std_per_group');

fprintf('Saved recovery results (Mode A, closed-form).\n');
