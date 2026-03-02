%% generate_T_PSD_3000x100_modeA_fixedI0.m
% -------------------------------------------------------------------------
% 用途
% - 物理仿真生成光谱/PSD 序列数据：每个参数组 (Delta_alpha, beta) 重复测量 Nmeas 次，
%   得到带噪声的测量谱 Tmeas，并汇总每组的均值/方差，同时可保存全部原始测量 Traw。
% - 模式 A：固定 I0（总入射功率）为常量，不做“每组总功率=常量”的标定。
%
% 主要输入（在脚本顶部直接改变量）
% - seed                : 随机种子（影响参数采样与噪声）
% - outdir              : 输出目录（不存在会 mkdir）
% - lambda (nm)         : 波长采样点，长度 L（当前 L=89）
% - N                   : 参数组数（注意：当前代码里 N=1000，但文件名写 3000；如需 3000 请改 N=3000）
% - Nmeas               : 每组重复测量次数（当前 Nmeas=100）
% - Delta_alpha, beta   : 每组真实参数（代码中随机采样范围）
% - I0                  : 固定功率（W），当前 I0=10e-3
% - Tint, eta           : 积分时间/量子效率（用于噪声建模）
% - 噪声参数：
%   * sigmaRIN_common   : common-mode RIN（整条谱共同缩放）
%   * sigmaRIN_shape    : shape RIN（随波长变化的乘性噪声）
%   * Lc_nm             : shape RIN 的相关长度（nm）
%   * bg_ratio          : 背景功率比例（Ibg/Isig）
%
% 关键输出（写入 outdir）
% 1) Excel：T_PSD_3000x100_ModeA_fixedI0_mean_std.xlsx
%    - Sheet 'T_mean' : [lambda_nm, mean_sample_0001 ... mean_sample_N]
%    - Sheet 'T_std'  : [lambda_nm, std_sample_0001  ... std_sample_N]
%    - Sheet 'params' : 每组参数与噪声/测量配置（Delta_alpha, beta, I0, Tint, eta, ...）
% 2) MAT（v7.3）：Traw_3000x100_PSD_ModeA_fixedI0_withNoise.mat
%    - 保存变量：lambda, omega, Delta_alpha, beta, I0, Tint, eta,
%              sigmaRIN_common, sigmaRIN_shape, Lc_nm, Traw
% 3) TXT：run_params_时间戳.txt
%    - 记录本次运行的关键参数（便于复现实验）
%
% 数据形状约定（MATLAB 维度）
% - lambda : [L, 1]
% - Delta_alpha / beta : [N, 1]
% - Traw   : [L, Nmeas, N]   （代码中使用 Traw(:, k, i)）
% - Tmean/Tstd: [L, N]
%
% 下游衔接
% - 该脚本输出的 .mat 会被 mat_to_npz_group_split.py 读取并转为 NPZ，
%   然后 resnet_main_0130.py 使用 train.npz/test.npz 训练网络。
% -------------------------------------------------------------------------
clc; clear;

%% ===== 0) 常数与网格 =====
%% ===== RUN LOG: 每次运行保存参数到txt（新增）=====
seed = 42;                      % 你可改：用于可复现实验
rng(seed, 'twister');           % 固定随机性（Delta_alpha/beta、RIN、shot noise 都会随 seed 变化）

run_tag  = datestr(now,'yyyymmdd_HHMMSS');
outdir = 'F:\DL\Code\07_ResNet1D\data_generate_matlab\20260302';

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

param_txt = fullfile(outdir , ['run_params_' run_tag '.txt']);


c    = 3e8;
tau  = 2.18e-16;
hbar = 1.054571817e-34;

lambda   = linspace(605.44, 645.36, 89).';   % nm (列向量)
lambda_m = lambda * 1e-9;                    % m

omega  = 2*pi*c ./ lambda_m;                 % rad/s
omega0 = 2*pi*c / (623.61e-9);
delta  = 2*pi*c*(5e-9)/(623.61e-9)^2;        % rad/s
alpha0 = tau * omega0;

den = 2*sqrt(pi)*abs(delta);                 % 2*sqrt(pi)*sqrt(delta^2)

bg_ratio = 0.10;   % Ibg/Isig：背景功率 = 信号总功率的 10%（定性分析很常用）

% 非均匀omega网格：每点bin宽度 Δω
domega = abs(gradient(omega));               % rad/s (89×1)

%% ===== 1) 参数组（3000组） =====
N = 1000;
Delta_alpha = -0.003 + 0.006*rand(N,1);      % [-0.003, 0.003]
beta        =  0.003*rand(N,1);              % [0, 0.003]

fprintf('Delta_alpha range: [%g, %g]\n', min(Delta_alpha), max(Delta_alpha));
fprintf('beta range:        [%g, %g]\n', min(beta), max(beta));

%% ===== 2) 噪声/测量参数（你可改） =====
I0 = 10e-3;         % 固定 I0 = 10 mW (W)，不再做每组总功率标定

Tint = 1e-3;        % 积分时间(s) 1e-3 origin
eta  = 0.5;         % 等效量子效率(0~1)

% RIN参数：二选一/可叠加
sigmaRIN_common = 0.00;   % common-mode RIN rms（整条谱同缩放），例如0.01=1%
sigmaRIN_shape  = 0.02;   % shape RIN rms（随波长变化），例如0.02=2%
Lc_nm = 3;                % shape RIN相关长度（nm），建议 1~10 nm

Nmeas = 100;        % 每组测量次数
nLam  = numel(lambda);

% 将相关长度从 nm 换成 “bin”
dlam_nm = mean(diff(lambda));               % nm/bin
Lc_bin  = max(1, round(Lc_nm / dlam_nm));   % 至少1

% shape RIN 平滑核（高斯核）
m = (-3*Lc_bin):(3*Lc_bin);
ker = exp(-(m.^2)/(2*(Lc_bin^2)));
ker = ker / sum(ker);

%% ===== 3) 结果存储 =====
Tmean = zeros(nLam, N);
Tstd  = zeros(nLam, N);

saveRaw = true;
if saveRaw
    % 89×100×3000，single约 107MB
    Traw = zeros(nLam, Nmeas, N, 'single');
end

%% ===== 4) 主循环：固定I0生成谱 + 噪声测量 =====
for i = 1:N
    alpha = alpha0 + Delta_alpha(i);

    % 未乘I0的谱形状 G(ω)，单位约 1/(rad/s)
    G = exp(-(omega - omega0).^2 / delta^2) .* ...
        ( -cos(2*(alpha - tau*omega)) + cosh(2*beta(i)) ) ./ den;

    % PSD物理保护（用于泊松均值）
    G = max(G, 0);

    % 模式A：固定 I0，不做每组总功率标定
    T0 = I0 * G;                           % W/(rad/s)
    T0 = max(T0, 0);

    % 在线均值/方差（Welford）
    mu = zeros(nLam,1);
    M2 = zeros(nLam,1);

    for k = 1:Nmeas
        % ===== 1) common-mode RIN（可选） =====
        epsC = sigmaRIN_common * randn();

        % ===== 2) shape RIN：波长相关乘性噪声（谱形状抖动） =====
        if sigmaRIN_shape > 0
            z = randn(nLam,1);
            epsLam = conv(z, ker, 'same');
            epsLam = epsLam - mean(epsLam);
            sdev = std(epsLam);
            if sdev > 0
                epsLam = epsLam / sdev;
            end
            epsLam = sigmaRIN_shape * epsLam;   % rms = sigmaRIN_shape
        else
            epsLam = zeros(nLam,1);
        end

        % 合成“真实PSD”
        % ===== 背景：按信号功率比例设置（每条曲线自适应难度）=====
        Isig = sum(T0 .* domega);                          % 信号总功率(W)
        Ibg  = bg_ratio * Isig;                            % 背景总功率(W)
        Tbg  = (Ibg / sum(domega)) * ones(nLam, 1);        % 平坦背景PSD：W/(rad/s)

        Ttrue = T0 .* (1 + epsC) .* (1 + epsLam) + Tbg;
        Ttrue = max(Ttrue, 0);

        % ===== 3) shot noise：PSD -> 光子数 -> Poisson -> PSD =====
        mu_ph = eta .* (Ttrue .* domega .* Tint) ./ (hbar .* omega);
        mu_ph = max(mu_ph, 0);

        % 需要 Statistics and Machine Learning Toolbox
        Nph = poissrnd(mu_ph);

        Tmeas = (Nph .* (hbar .* omega)) ./ (eta .* Tint .* domega);
        Tmeas = max(Tmeas, 0);

        if saveRaw
            Traw(:,k,i) = single(Tmeas);
        end

        % 在线均值/方差
        d1 = Tmeas - mu;
        mu = mu + d1 / k;
        d2 = Tmeas - mu;
        M2 = M2 + d1 .* d2;
    end

    Tmean(:,i) = mu;
    Tstd(:,i)  = sqrt(M2 / max(Nmeas-1,1));

    if mod(i,200)==0
        % 监测：总功率不再恒定，会随参数变
        Pcheck = sum(T0 .* domega);
        fprintf('Processed %d/%d  (example total power ~ %.3e W)\n', i, N, Pcheck);
    end
end

%% ===== 5) 写入 Excel（mean/std + params） =====
outFile = strcat(outdir,'\','T_PSD_3000x100_ModeA_fixedI0_mean_std.xlsx');

% mean sheet
dataMean = [lambda, Tmean];
headerMean = cell(1, N+1);
headerMean{1} = 'lambda_nm';
for i = 1:N
    headerMean{i+1} = sprintf('mean_sample_%04d', i);
end
writecell(headerMean, outFile, 'Sheet', 'T_mean', 'Range', 'A1');
writematrix(dataMean, outFile, 'Sheet', 'T_mean', 'Range', 'A2');

% std sheet
dataStd = [lambda, Tstd];
headerStd = cell(1, N+1);
headerStd{1} = 'lambda_nm';
for i = 1:N
    headerStd{i+1} = sprintf('std_sample_%04d', i);
end
writecell(headerStd, outFile, 'Sheet', 'T_std', 'Range', 'A1');
writematrix(dataStd, outFile, 'Sheet', 'T_std', 'Range', 'A2');

% params sheet（记录I0与噪声参数）
paramHeader = {'sample_id','Delta_alpha','beta','I0_W','Tint_s','eta', ...
               'sigmaRIN_common','sigmaRIN_shape','Lc_nm','Nmeas'};
paramData = [(1:N)', Delta_alpha, beta, ...
             I0*ones(N,1), Tint*ones(N,1), eta*ones(N,1), ...
             sigmaRIN_common*ones(N,1), sigmaRIN_shape*ones(N,1), Lc_nm*ones(N,1), ...
             Nmeas*ones(N,1)];
writecell(paramHeader, outFile, 'Sheet', 'params', 'Range', 'A1');
writematrix(paramData, outFile, 'Sheet', 'params', 'Range', 'A2');

fprintf('Done. Saved mean/std to %s\n', outFile);

%% ===== 6) 保存全部原始测量到 MAT =====
if saveRaw
    outMat = strcat(outdir,'\','Traw_3000x100_PSD_ModeA_fixedI0_withNoise.mat');
    save(outMat, 'lambda', 'omega', 'Delta_alpha', 'beta', ...
         'I0', 'Tint', 'eta', 'sigmaRIN_common', 'sigmaRIN_shape', 'Lc_nm', ...
         'Traw', '-v7.3');
    fprintf('Raw measurements saved to %s\n', outMat);
end

%% ===== 7) 保存本次运行参数到 txt（新增）=====
params = struct();
params.timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS');
params.seed = seed;
params.matlab_version = version;

% 常数/网格
params.c = c;
params.tau = tau;
params.hbar = hbar;
params.omega0 = omega0;
params.delta = delta;
params.den = den;

params.lambda_nm_first = lambda(1);
params.lambda_nm_last  = lambda(end);
params.lambda_nm_len   = numel(lambda);
params.dlam_nm_mean    = dlam_nm;

% 数据规模
params.N = N;
params.Nmeas = Nmeas;
params.nLam = nLam;

% 测量/噪声
params.I0_W = I0;
params.Tint_s = Tint;
params.eta = eta;

params.sigmaRIN_common = sigmaRIN_common;
params.sigmaRIN_shape  = sigmaRIN_shape;
params.Lc_nm = Lc_nm;
params.Lc_bin = Lc_bin;

% 背景
params.bg_ratio = bg_ratio;

% 输出文件
params.out_excel = outFile;
params.saveRaw = saveRaw;
params.out_mat = outMat;

save_params_txt(params, param_txt);
fprintf('[OK] Saved params txt: %s\n', param_txt);

%% ===== local function：写 key=value 到 txt（新增）=====
function save_params_txt(p, path)
    fid = fopen(path, 'w');
    assert(fid > 0, 'Cannot open: %s', path);

    fns = fieldnames(p);
    for i = 1:numel(fns)
        k = fns{i};
        v = p.(k);

        if isnumeric(v)
            if isscalar(v)
                fprintf(fid, '%s = %.12g\n', k, v);
            else
                fprintf(fid, '%s = %s\n', k, mat2str(v));
            end
        elseif isstring(v) || ischar(v)
            fprintf(fid, '%s = %s\n', k, string(v));
        else
            try
                fprintf(fid, '%s = %s\n', k, evalc('disp(v)'));
            catch
                fprintf(fid, '%s = [unprintable]\n', k);
            end
        end
    end

    fclose(fid);
end