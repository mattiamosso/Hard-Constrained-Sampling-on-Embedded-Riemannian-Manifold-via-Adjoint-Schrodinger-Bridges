% Extended R-ASBS (Alg. 2) on the Stiefel manifold V(4,2): expected energy vs beta,
% compared with geodesic Monte Carlo, Riemannian ULA and the exact reference.
% ("ASBS-M" in the code = R-ASBS / Extended R-ASBS in the paper)

clear; clc; close all;
% Inverse temperatures
Beta = [0.001, 0.01, 0.1, 0.5, 1.3, 2, 5, 7, 10, 20, 50, 100, 200, 1000, 10000, 1000000];

empirical_energy = zeros(length(Beta), 1);

% Metric and baseline settings
rng(0, 'twister');
run_baselines = true;       % run the MCMC baselines
save_results  = true;       % save .csv / .mat

% Geodesic Monte Carlo (Byrne & Girolami, 2013)
gmc_opts.n_warmup   = 200;  % warm-up (step adapted)
gmc_opts.n_iter     = 300;
gmc_opts.L          = 10;   % leapfrog steps
gmc_opts.target_acc = 0.75;
gmc_opts.eps_max    = 0.25;

% Riemannian ULA, same gradient budget per chain as GMC
rula_opts.n_steps = (gmc_opts.n_warmup + gmc_opts.n_iter) * gmc_opts.L;
rula_opts.n_burn  = gmc_opts.n_warmup * gmc_opts.L;
rula_opts.thin    = gmc_opts.L;
rula_opts.c_h     = 0.05;         % h = c_h / (2 beta (lam_max - lam_min))
rula_opts.h_max   = 1e-2;

% ASBS-M options
asbs_opts.N_steps   = 100;    % Euler-retraction steps
asbs_opts.sigma_max = 2.0;    % sigma(t) = sigma_max^(1-t) sigma_min^t
asbs_opts.sigma_min = 0.1;    % capped by 1/sqrt(beta (lam_max - lam_min))
asbs_opts.K_epochs  = 400;
asbs_opts.B         = 600;    % trajectories per epoch
asbs_opts.n_inner   = 20;     % Adam steps per epoch
asbs_opts.n_sub     = 10;     % minibatch = n_sub*B triples
asbs_opts.lr        = 1e-3;
asbs_opts.lr_end    = 1;      % cosine decay factor (1 = constant)
asbs_opts.use_corrector = true;   % chord corrector h
asbs_opts.adjoint = 'projection'; % 'projection' = Alg. 2; 'exact' = VJP (optional)
asbs_opts.loss_weight = true;     % 1/sigma_t^2 loss weight (false = unweighted)
asbs_opts.select_by_cost = false; % optional checkpoint selection (not in the paper)
asbs_opts.val_every = 25;
asbs_opts.n_val     = 2000;

% GMC with the per-sample budget of ASBS-M
gmcb_opts = gmc_opts;
n_budget  = ceil(asbs_opts.N_steps / gmc_opts.L);
gmcb_opts.n_warmup = floor(n_budget/2);
gmcb_opts.n_iter   = n_budget - gmcb_opts.n_warmup;

method_names = {'ASBS-M', 'GMC', 'RULA', 'GMC (same budget)'};
nB = length(Beta);
nM = numel(method_names);
res = repmat(empty_metrics(), nB, nM);  % res(beta, method)
E_ref    = nan(nB, 1);                  % exact E_pi[tr(X'HX)]
E_sd_ref = nan(nB, 1);                  % exact sd
t_train  = nan(nB, 1);
hist_all  = cell(nB, 1);
gmc_info  = cell(nB, 1);
rula_info = cell(nB, 1);
gmcb_info = cell(nB, 1);

% Warm start across betas
netU_prev = [];
netH_prev = [];

% Loop over beta
for beta_idx = 1:length(Beta)
    beta = Beta(beta_idx);
    fprintf('Running beta = %.5f\n', beta);

    t_start = tic;
    [netU, netH, params, train_hist] = ASBS_Stiefel_sampler(beta, netU_prev, netH_prev, asbs_opts);
    t_train(beta_idx) = toc(t_start);
    hist_all{beta_idx} = train_hist;

    netU_prev = netU;
    netH_prev = netH;

    % Sampling from Haar initial points
    number_samples = 5000;
    X0 = zeros(params.n, params.p, number_samples);
    for i = 1:number_samples
        X0(:,:,i) = Retraction(randn(params.n, params.p));
    end

    t_start = tic;
    final_samples = simulate_trajectories(netU, X0, params.sig, params.dt, ...
        params.N_steps, number_samples, params.n, params.p);
    t_sample = toc(t_start);
    X_final = final_samples(:,:,:, params.N_steps+1);
    clear final_samples

    % Empirical energy
    for i = 1:number_samples
        empirical_energy(beta_idx) = empirical_energy(beta_idx) + trace(X_final(:,:,i)' * params.H * X_final(:,:,i))';
    end
    empirical_energy(beta_idx) = empirical_energy(beta_idx) / number_samples;

    fprintf('Theoretical Minimum Energy (sum of lowest %d evals): %.2f\n', params.p, params.target_min_energy);
    fprintf('Trained Model Expected Energy E[tr(X^T H X)]: %.4f\n \n', empirical_energy(beta_idx));

    % Exact reference and metrics
    ref = exact_reference_Stiefel42(params.H, params.p, beta);
    E_ref(beta_idx)    = ref.mean;
    E_sd_ref(beta_idx) = sqrt(ref.var);

    m = sample_metrics(X_final, params.H, ref);
    m.evals_per_sample = params.N_steps;
    m.time_s           = t_sample;
    m.train_time_s     = t_train(beta_idx);
    res(beta_idx, 1) = m;

    % Baselines from the same Haar points
    if run_baselines
        % Geodesic Monte Carlo
        t_start = tic;
        [X_gmc, info] = gmc_stiefel(X0, params.H, beta, gmc_opts);
        t_gmc = toc(t_start);
        m = sample_metrics(X_gmc, params.H, ref);
        m.evals_per_sample = info.grad_evals;
        m.time_s    = t_gmc;
        m.acc_rate  = info.acc_rate;
        m.step_size = info.step_size;
        [m.rhat, m.ess] = ess_rhat(info.E_trace);
        res(beta_idx, 2) = m;
        gmc_info{beta_idx} = rmfield(info, 'E_trace');

        % Riemannian ULA
        t_start = tic;
        [X_rula, info] = rula_stiefel(X0, params.H, beta, rula_opts);
        t_rula = toc(t_start);
        m = sample_metrics(X_rula, params.H, ref);
        m.evals_per_sample = info.grad_evals;
        m.time_s    = t_rula;
        m.step_size = info.step_size;
        [m.rhat, m.ess] = ess_rhat(info.E_trace);
        res(beta_idx, 3) = m;
        rula_info{beta_idx} = rmfield(info, 'E_trace');

        % GMC with the budget of ASBS-M
        t_start = tic;
        [X_gb, info] = gmc_stiefel(X0, params.H, beta, gmcb_opts);
        t_gb = toc(t_start);
        m = sample_metrics(X_gb, params.H, ref);
        m.evals_per_sample = info.grad_evals;
        m.time_s    = t_gb;
        m.acc_rate  = info.acc_rate;
        m.step_size = info.step_size;
        res(beta_idx, 4) = m;
        gmcb_info{beta_idx} = rmfield(info, 'E_trace');
    end

    fprintf('Exact reference E_pi[tr(X^T H X)] = %.6f  (sd %.3e)\n', ref.mean, sqrt(ref.var));
    for k = 1:nM
        r = res(beta_idx, k);
        if isnan(r.E_hat), continue; end
        fprintf('  %-17s E = %.6f | |bias| = %.2e (z = %6.2f) | KS = %.4f | occTV = %.4f | orth = %.1e', ...
            method_names{k}, r.E_hat, r.abs_bias, r.bias_z, r.KS, r.occ_TV, r.orth_err);
        fprintf(' | orbit Kuiper = %.3f | det imb. = %.3f', r.orbit_kuiper, r.det_imbalance);
        if ~isnan(r.rhat),     fprintf(' | Rhat = %.3f | ESS = %.0f', r.rhat, r.ess); end
        if ~isnan(r.acc_rate), fprintf(' | acc = %.2f', r.acc_rate); end
        fprintf(' | %.1f s\n', r.time_s);
    end
    fprintf('  (KS 95%% threshold for N = %d: %.4f; Kuiper: %.4f; det imbalance 2 s.d.: %.4f)\n\n', ...
        number_samples, 1.36/sqrt(number_samples), 1.747/sqrt(number_samples), 1/sqrt(number_samples));
end

% Exact curve on a dense beta grid
beta_dense = logspace(log10(min(Beta)), log10(max(Beta)), 200);
E_dense = zeros(size(beta_dense));
for i = 1:numel(beta_dense)
    ref_i = exact_reference_Stiefel42(params.H, params.p, beta_dense(i));
    E_dense(i) = ref_i.mean;
end

%% Plot
fig = figure('Units', 'inches', 'Position', [2, 2, 6, 4.5], 'Color', 'w');

c_navy     = [0, 51, 102] / 255;
c_burgundy = [153, 51, 51] / 255;
c_grey     = [102, 102, 102] / 255;
c_green    = [0, 160, 122] / 255;
c_ochre    = [204, 121, 0] / 255;

hold on;

p1 = plot(Beta, empirical_energy, '-o', 'Color', c_navy, ...
    'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', [0.8, 0.9, 1.0]);

p2 = plot(Beta, ones(size(Beta)) * params.target_min_energy, '--', ...
    'Color', c_burgundy, 'LineWidth', 2.5);

p3 = plot(Beta, ones(size(Beta)) * params.target_max_energy, '--', ...
    'Color', c_grey, 'LineWidth', 2.5);

p4 = plot(beta_dense, E_dense, '-', 'Color', 'k', 'LineWidth', 1.5);
c_olive    = [120, 120, 40] / 255;
if run_baselines
    p5 = plot(Beta, [res(:,2).E_hat], '-s', 'Color', c_green, ...
        'LineWidth', 2, 'MarkerSize', 7, 'MarkerEdgeColor', 'k', ...
        'MarkerFaceColor', 0.6*c_green + 0.4);
    p6 = plot(Beta, [res(:,3).E_hat], '-^', 'Color', c_ochre, ...
        'LineWidth', 2, 'MarkerSize', 7, 'MarkerEdgeColor', 'k', ...
        'MarkerFaceColor', 0.6*c_ochre + 0.4);
end
uistack(p1, 'top');

ax = gca;
ax.XScale = 'log';
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.LineWidth = 1.2;
ax.FontSize = 20;
ax.FontName = 'Times New Roman';
ax.Box = 'on';
grid on;
ax.GridAlpha = 0.2;

xlabel('Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Expected Energy $\mathbb{E}[\mathrm{Tr}(X^{\top} H X)]$', 'Interpreter', 'latex', 'FontSize', 16);
title('Convergence to Low Energy Subspace', 'Interpreter', 'latex', 'FontSize', 16);

if run_baselines
    leg = legend([p1, p5, p6, p4, p2, p3], {'ASBS-M', 'GMC', 'RULA', 'Exact', 'Min', 'Max'}, ...
        'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
else
    leg = legend([p1, p4, p2, p3], {'ASBS-M', 'Exact', 'Min', 'Max'}, ...
        'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
end
leg.Box = 'off';

%% Accuracy metrics vs beta
cols = {c_navy, c_green, c_ochre, c_olive};
mks  = {'o', 's', '^', 'v'};
if run_baselines, active = 1:nM; else, active = 1; end
N_eval = number_samples;

fig2 = figure('Units', 'inches', 'Position', [2, 2, 15, 8], 'Color', 'w');
tl2 = tiledlayout(fig2, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl2); hold(ax, 'on');
hs = [];
hs(end+1) = plot(ax, beta_dense, E_dense - params.target_min_energy, 'k-', 'LineWidth', 1.5);
bl = beta_dense(beta_dense >= 1);
hs(end+1) = plot(ax, bl, params.p*(params.n - params.p) ./ (2*bl), ':', 'Color', c_grey, 'LineWidth', 2);
for k = active
    hs(end+1) = plot(ax, Beta, [res(:,k).E_hat] - params.target_min_energy, ['-' mks{k}], ...
        'Color', cols{k}, 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
legend(hs, [{'Exact', '$p(n-p)/(2\beta)$'}, method_names(active)], 'Interpreter', 'latex', ...
    'Location', 'southwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
ylabel(ax, '$\mathbb{E}[\mathrm{Tr}(X^{\top}HX)] - E_{\min}$', 'Interpreter', 'latex');
title(ax, '(a) Excess energy', 'Interpreter', 'latex');

ax = nexttile(tl2); hold(ax, 'on');
hs = [];
for k = active
    hs(end+1) = plot(ax, Beta, [res(:,k).abs_bias], ['-' mks{k}], 'Color', cols{k}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
hs(end+1) = plot(ax, Beta, 2*E_sd_ref/sqrt(N_eval), '--', 'Color', c_grey, 'LineWidth', 1.5);
legend(hs, [method_names(active), {'$2\sigma_\pi/\sqrt{N}$'}], 'Interpreter', 'latex', ...
    'Location', 'southwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
ylabel(ax, '$|\hat{\mathbb{E}}[\mathrm{Tr}(X^{\top}HX)] - \mathbb{E}_\pi[\cdot]|$', 'Interpreter', 'latex');
title(ax, '(b) Bias of mean energy', 'Interpreter', 'latex');

ax = nexttile(tl2); hold(ax, 'on');
hs = [];
for k = active
    hs(end+1) = plot(ax, Beta, [res(:,k).KS], ['-' mks{k}], 'Color', cols{k}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
hs(end+1) = plot(ax, [min(Beta), max(Beta)], 1.36/sqrt(N_eval)*[1, 1], '--', 'Color', c_grey, 'LineWidth', 1.5);
legend(hs, [method_names(active), {'95\% KS threshold'}], 'Interpreter', 'latex', ...
    'Location', 'northwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
xlabel(ax, 'Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex');
ylabel(ax, '$\sup_e |\hat F(e) - F_\pi(e)|$', 'Interpreter', 'latex');
title(ax, '(c) KS distance of energy law', 'Interpreter', 'latex');

ax = nexttile(tl2); hold(ax, 'on');
hs = [];
for k = active
    hs(end+1) = plot(ax, Beta, [res(:,k).occ_TV], ['-' mks{k}], 'Color', cols{k}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
legend(hs, method_names(active), 'Interpreter', 'latex', 'Location', 'northwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
xlabel(ax, 'Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex');
ylabel(ax, '$\frac{1}{2p}\sum_i |\hat m_i - m_i^\pi|$', 'Interpreter', 'latex');
title(ax, '(d) Spectral occupation error', 'Interpreter', 'latex');

% O(2)-symmetry checks
ax = nexttile(tl2); hold(ax, 'on');
hs = [];
for k = active
    hs(end+1) = plot(ax, Beta, [res(:,k).orbit_kuiper], ['-' mks{k}], 'Color', cols{k}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
hs(end+1) = plot(ax, [min(Beta), max(Beta)], 1.747/sqrt(N_eval)*[1, 1], '--', 'Color', c_grey, 'LineWidth', 1.5);
legend(hs, [method_names(active), {'95\% threshold'}], 'Interpreter', 'latex', 'Location', 'northwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
xlabel(ax, 'Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex');
ylabel(ax, 'Kuiper statistic of the $O(2)$ angle', 'Interpreter', 'latex');
title(ax, '(e) Coverage of the $O(2)$ orbit', 'Interpreter', 'latex');

ax = nexttile(tl2); hold(ax, 'on');
hs = [];
for k = active
    hs(end+1) = plot(ax, Beta, max([res(:,k).det_imbalance], 1e-5), ['-' mks{k}], 'Color', cols{k}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', cols{k});
end
hs(end+1) = plot(ax, [min(Beta), max(Beta)], 1/sqrt(N_eval)*[1, 1], '--', 'Color', c_grey, 'LineWidth', 1.5);
legend(hs, [method_names(active), {'2 s.d.'}], 'Interpreter', 'latex', 'Location', 'northwest', 'Box', 'off');
format_metric_axes(ax, 'loglog');
xlabel(ax, 'Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex');
ylabel(ax, '$|\hat P(\det Q > 0) - 1/2|$', 'Interpreter', 'latex');
title(ax, '(f) Balance of the two components', 'Interpreter', 'latex');

%% Cost vs accuracy
beta_show = Beta(ismember(Beta, [0.5, 1.3, 10, 100]));
if isempty(beta_show), beta_show = Beta(unique(round(linspace(1, nB, min(nB, 4))))); end
fig3 = figure('Units', 'inches', 'Position', [2, 2, 3.2*numel(beta_show), 3.6], 'Color', 'w');
tl3 = tiledlayout(fig3, 1, numel(beta_show), 'TileSpacing', 'compact', 'Padding', 'compact');
for b = beta_show
    i = find(Beta == b, 1);
    ax = nexttile(tl3); hold(ax, 'on');
    hs = [];
    if run_baselines && ~isempty(gmc_info{i})
        Tg = numel(gmc_info{i}.E_mean);
        hs(end+1) = plot(ax, (1:Tg)*gmc_opts.L, abs(gmc_info{i}.E_mean - E_ref(i)), '-', ...
            'Color', c_green, 'LineWidth', 1.5);
    end
    hs(end+1) = plot(ax, asbs_opts.N_steps, res(i,1).abs_bias, 'o', 'Color', c_navy, ...
        'MarkerFaceColor', c_navy, 'MarkerSize', 8);
    hs(end+1) = plot(ax, [1, 1e4], 2*E_sd_ref(i)/sqrt(N_eval)*[1, 1], '--', 'Color', c_grey, 'LineWidth', 1.2);
    format_metric_axes(ax, 'loglog');
    xlim(ax, [1, 1e4]);
    xlabel(ax, 'Model evaluations per sample', 'Interpreter', 'latex');
    if b == beta_show(1), ylabel(ax, '$|\hat{\mathbb{E}}[\mathrm{Tr}(X^{\top}HX)] - \mathbb{E}_\pi[\cdot]|$', 'Interpreter', 'latex'); end
    title(ax, sprintf('$\\beta = %g$', b), 'Interpreter', 'latex');
end
if run_baselines
    legend(hs, {'GMC from Haar', 'ASBS-M', '$2\sigma_\pi/\sqrt{N}$'}, ...
        'Interpreter', 'latex', 'Location', 'southwest', 'Box', 'off');
else
    legend(hs, {'ASBS-M', '$2\sigma_\pi/\sqrt{N}$'}, ...
        'Interpreter', 'latex', 'Location', 'southwest', 'Box', 'off');
end

%% Summary table and saving
var_names = {'beta', 'method', 'E_hat', 'E_exact', 'abs_bias', 'bias_z', 'KS', 'occ_TV', ...
    'orth_err', 'rhat', 'ess', 'ess_per_s', 'acc_rate', 'step_size', 'evals_per_sample', ...
    'time_s', 'train_time_s', 'orbit_kuiper', 'det_imbalance'};
T_res = cell(0, numel(var_names));
for i = 1:nB
    for k = active
        r = res(i, k);
        T_res(end+1, :) = {Beta(i), method_names{k}, r.E_hat, E_ref(i), r.abs_bias, r.bias_z, ...
            r.KS, r.occ_TV, r.orth_err, r.rhat, r.ess, r.ess / r.time_s, r.acc_rate, ...
            r.step_size, r.evals_per_sample, r.time_s, r.train_time_s, r.orbit_kuiper, r.det_imbalance};
    end
end
T_res = cell2table(T_res, 'VariableNames', var_names);
disp(T_res);
if save_results
    writetable(T_res, 'alg2_stiefel_metrics.csv');
    save('alg2_stiefel_results.mat', 'Beta', 'res', 'E_ref', 'E_sd_ref', 'beta_dense', 'E_dense', ...
        'hist_all', 'gmc_info', 'rula_info', 'gmc_opts', 'rula_opts', 'T_res', ...
        'gmcb_info', 'asbs_opts', 'gmcb_opts');
end

%% Helper functions

function [netU, netH, params, train_hist] = ASBS_Stiefel_sampler(beta, netU_prev, netH_prev, aopt)
    params.n = 4;
    params.p = 2;
    params.B = aopt.B;

    params.N_steps  = aopt.N_steps;
    params.lr       = aopt.lr;
    params.dt       = 1/params.N_steps;
    params.K_epochs = aopt.K_epochs;

    params.H = [4,   0.5, 2.5, 1;
                0.5, 4,   1,   2.5;
                2.5, 1,   4,   0.5;
                1,   2.5, 0.5, 4];

    spec_H = eig(params.H);
    ordered_spec = sort(spec_H, 'ascend');
    params.target_min_energy = sum(ordered_spec(1:params.p));
    params.target_max_energy = params.p / params.n * sum(spec_H);

    % Geometric sigma(t); sigma_min capped by the target width
    s_min = min(aopt.sigma_min, 1/sqrt(beta*(ordered_spec(end) - ordered_spec(1))));
    t_grid = (0:params.N_steps-1)' * params.dt;
    params.sig   = aopt.sigma_max.^(1 - t_grid) .* s_min.^t_grid;
    params.sigma = params.sig;
    params.V1    = sum(params.sig.^2) * params.dt;

    % Networks: input = entries of XX' (+ t), output = n x n matrix F, u = F X
    n_feat = params.n*(params.n + 1)/2;
    layersU = [
        featureInputLayer(n_feat + 1, 'Normalization', 'none', 'Name', 'in_u')
        fullyConnectedLayer(512, 'Name', 'fc1_u')
        layerNormalizationLayer('Name', 'ln1_u')
        geluLayer('Name', 'gelu1_u')
        fullyConnectedLayer(512, 'Name', 'fc2_u')
        layerNormalizationLayer('Name', 'ln2_u')
        geluLayer('Name', 'gelu2_u')
        fullyConnectedLayer(params.n*params.n, 'Name', 'out_u')
        ];
    netU = dlnetwork(layersU);
    netU = initialize(netU);
    if ~isempty(netU_prev)
        try
            netU.Learnables = netU_prev.Learnables;
        catch
        end
    end

    layersH = [
        featureInputLayer(n_feat, 'Normalization', 'none', 'Name', 'in_h')
        fullyConnectedLayer(512, 'Name', 'fc1_h')
        layerNormalizationLayer('Name', 'ln1_h')
        geluLayer('Name', 'gelu1_h')
        fullyConnectedLayer(512, 'Name', 'fc2_h')
        layerNormalizationLayer('Name', 'ln2_h')
        geluLayer('Name', 'gelu2_h')
        fullyConnectedLayer(params.n*params.n, 'Name', 'out_h')
        ];
    netH = dlnetwork(layersH);
    netH = initialize(netH);
    if ~isempty(netH_prev)
        try
            netH.Learnables = netH_prev.Learnables;
        catch
        end
    end

    [avgU, avgSqU, avgH, avgSqH] = deal([]);
    iterU = 0;  iterH = 0;

    train_hist.lossU  = zeros(params.K_epochs, 1);
    train_hist.lossH  = zeros(params.K_epochs, 1);
    train_hist.energy = zeros(params.K_epochs, 1);

    fprintf('Starting ASBS-M Training (Beta = %.5f, N_steps = %d, lr = %.1e, sigma: %.2f -> %.3f)...\n', ...
        beta, params.N_steps, params.lr, params.sig(1), s_min);

    % Optional checkpoint selection on a fixed validation batch (not in the paper)
    if aopt.select_by_cost
        rng_state = rng;  rng(12345);
        Xv0 = batchRetraction(randn(params.n, params.p, aopt.n_val));
        XIv = randn(params.n, params.p, aopt.n_val, params.N_steps);
        rng(rng_state);
        best_J = inf;  best_epoch = 0;  netU_best = netU;  netH_best = netH;
        train_hist.J = nan(params.K_epochs, 1);
    end

    M_all = params.B * params.N_steps;
    M_sub = min(M_all, aopt.n_sub * params.B);
    T_traj = kron(t_grid', ones(1, params.B));
    S_traj = kron(params.sig', ones(1, params.B));
    for k = 1:params.K_epochs
        lr_k = params.lr * (aopt.lr_end + (1 - aopt.lr_end)*0.5*(1 + cos(pi*(k-1)/params.K_epochs)));
        X0 = batchRetraction(randn(params.n, params.p, params.B));   % Haar (QR, diag(R) > 0)

        [X_traj, XI_traj, A_traj] = simulate_trajectories(netU, X0, params.sig, params.dt, ...
            params.N_steps, params.B, params.n, params.p);

        X_N = X_traj(:,:,:,params.N_steps+1);
        h_val = predict_net(netH, X_N, params.n, params.p, params.B, false);
        if ~aopt.use_corrector, h_val = 0; end

        gradE = 2 * beta * pagemtimes(params.H, X_N);
        v_curr = batchProjector(X_N, gradE + h_val);       % P(grad E + h), as in Alg. 2

        V_traj = zeros(params.n, params.p, params.B, params.N_steps);
        for j = params.N_steps:-1:1
            X_j = X_traj(:,:,:,j);
            if strcmp(aopt.adjoint, 'exact')                % 'exact': VJP of step j
                V_traj(:,:,:,j) = step_vjp(X_j, X_traj(:,:,:,j+1), A_traj(:,:,:,j), v_curr, ...
                    params.sig(j)*sqrt(params.dt), XI_traj(:,:,:,j));
            else                                            % Alg. 2: backward projection
                V_traj(:,:,:,j) = batchProjector(X_j, v_curr);
            end
            v_curr = V_traj(:,:,:,j);
        end

        X_traj_flat = reshape(X_traj(:,:,:,1:params.N_steps), params.n, ...
            params.p, M_all);
        V_traj_flat = reshape(V_traj, params.n, params.p, M_all);

        % n_inner Adam steps on random minibatches
        for it = 1:aopt.n_inner
            idx = randi(M_all, 1, M_sub);
            iterU = iterU + 1;
            [gradU, lossU] = dlfeval(@computeLossU, netU, X_traj_flat(:,:,idx), T_traj(idx), ...
                V_traj_flat(:,:,idx), reshape(S_traj(idx), 1, 1, []), params.n, params.p, M_sub, aopt.loss_weight);
            [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, iterU, lr_k);
        end

        lossH_val = NaN;
        if aopt.use_corrector
            X_resampled = simulate_trajectories(netU, X0, params.sig, params.dt, ...
                params.N_steps, params.B, params.n, params.p);
            X_1 = X_resampled(:,:,:,params.N_steps + 1);

            chord = (X_1 - X0) / params.V1;                 % total reference variance
            b_target = - batchProjector(X_1, chord);

            for it = 1:aopt.n_inner
                iterH = iterH + 1;
                [gradH, lossH] = dlfeval(@computeLossH, netH, X_1, b_target, params.n, params.p, params.B);
                [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, iterH, lr_k);
            end
            lossH_val = double(extractdata(lossH));
        else
            X_1 = X_N;
        end

        train_hist.lossU(k)  = double(extractdata(lossU));
        train_hist.lossH(k)  = lossH_val;
        train_hist.energy(k) = mean(sum(X_1 .* pagemtimes(params.H, X_1), [1 2]), 'all');

        if aopt.select_by_cost && mod(k, aopt.val_every) == 0 && k >= params.K_epochs/4
            train_hist.J(k) = control_cost(netU, Xv0, XIv, params.sig, params.dt, beta, params.H);
            if train_hist.J(k) < best_J
                best_J = train_hist.J(k);  best_epoch = k;  netU_best = netU;  netH_best = netH;
            end
        end

        if mod(k, 100) == 0 || k==1
            fprintf('Epoch %04d | Loss U: %.4f | Loss H: %.4f | E[tr(X''HX)] = %.4f\n', ...
                k, train_hist.lossU(k), lossH_val, train_hist.energy(k));
        end
    end
    if aopt.select_by_cost && best_epoch > 0
        netU = netU_best;  netH = netH_best;
        fprintf('Selected checkpoint: epoch %d (validation J = %.4f)\n', best_epoch, best_J);
    end
end

function J = control_cost(net, X0, XI, sigma, dt, beta, H)
    % J(u) = mean of 1/2 sum |u|^2 dt + beta tr(X1'HX1) on a fixed batch
    [n, p, B, N] = size(XI);
    X = X0;  c = zeros(1, 1, B);
    for j = 1:N
        u = batchProjector(X, predict_net(net, X, n, p, B, true, (j-1)*dt));
        c = c + 0.5*dt*sum(u.^2, [1 2]);
        X = batchRetraction(X + sigma(j)*u*dt + sigma(j)*sqrt(dt)*batchProjector(X, XI(:,:,:,j)));
    end
    J = mean(c(:) + beta*reshape(sum(X .* pagemtimes(H, X), [1 2]), [], 1));
end

function [gradients, loss] = computeLossU(netU, X_batch, t_batch, V_target, sigma, n, p, BatchSize, weighted)
    inputs = dlarray([equiv_features(X_batch); t_batch], 'CB');

    U_pred_flat = forward(netU, inputs);
    U_pred = equiv_output(U_pred_flat, X_batch);                     % u = F X

    X_dl = dlarray(X_batch);
    U_proj = batchProjector(X_dl, U_pred);
    V_dl = dlarray(V_target);

    residuals = U_proj + sigma .* V_dl;
    % Optional 1/sigma_t^2 weight (same minimiser)
    if weighted
        loss = sum((residuals ./ sigma).^2, 'all') / BatchSize;
    else
        loss = sum(residuals.^2, 'all') / BatchSize;
    end
    gradients = dlgradient(loss, netU.Learnables);
end

function [gradients, loss] = computeLossH(netH, X_N, b_target, n, p, BatchSize)
    inputs = dlarray(equiv_features(X_N), 'CB');

    H_pred_flat = forward(netH, inputs);
    H_pred = equiv_output(H_pred_flat, X_N);                         % h = G X

    X_dl = dlarray(X_N);
    H_proj = batchProjector(X_dl, H_pred);
    b_dl = dlarray(b_target);

    residuals = H_proj - b_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
    gradients = dlgradient(loss, netH.Learnables);
end

function [X_traj, XI, A_pre] = simulate_trajectories(net, X0, sigma, dt, N, B, n, p)
    X_traj = zeros(n, p, B, N+1);
    X_traj(:,:,:,1) = X0;
    keep = nargout > 1;                                     % store noises and pre-retraction points
    if keep, XI = zeros(n, p, B, N); A_pre = zeros(n, p, B, N); end

    for j = 1:N
        t = (j-1)*dt;
        s = sigma(min(j, numel(sigma)));
        X_curr = X_traj(:,:,:,j);

        u_val = predict_net(net, X_curr, n, p, B, true, t);
        noise = randn(n, p, B);

        drift = s * batchProjector(X_curr, u_val) * dt;
        diff_term= s * sqrt(dt) * batchProjector(X_curr, noise);
        step_update = X_curr + drift + diff_term;

        X_traj(:,:,:,j+1) = batchRetraction(step_update);
        if keep, XI(:,:,:,j) = noise; A_pre(:,:,:,j) = step_update; end
    end
end

function V = step_vjp(X, Q, A, G, c, xi)
    % Exact adjoint transport through one simulator step (VJP, drift held fixed)
    p  = size(X, 2);
    R  = pagemtimes(Q, 'transpose', A, 'none');
    M  = pagemtimes(Q, 'transpose', G, 'none');
    Lm = (M - permute(M, [2 1 3])) .* tril(ones(p), -1);
    Abar = pagemtimes(G - pagemtimes(Q, M) + pagemtimes(Q, Lm), permute(batch_triu_inv(R), [2 1 3]));
    S1 = pagemtimes(X, 'transpose', xi, 'none');   S1 = 0.5*(S1 + permute(S1, [2 1 3]));
    S2 = pagemtimes(X, 'transpose', Abar, 'none'); S2 = 0.5*(S2 + permute(S2, [2 1 3]));
    V  = batchProjector(X, Abar - c*(pagemtimes(Abar, S1) + pagemtimes(xi, S2)));
end

function Ri = batch_triu_inv(R)
    % Batched inverse of upper-triangular p x p matrices
    p  = size(R, 1);
    Ri = zeros(size(R));
    for i = p:-1:1
        Ri(i, i, :) = 1 ./ R(i, i, :);
        for j = i+1:p
            acc = zeros(1, 1, size(R, 3));
            for k = i+1:j
                acc = acc + R(i, k, :) .* Ri(k, j, :);
            end
            Ri(i, j, :) = -acc ./ R(i, i, :);
        end
    end
end

function out = predict_net(net, X, n, p, B, has_time, t)
    f = equiv_features(X);
    if has_time
        inputs = dlarray([f; repmat(t, 1, B)], 'CB');
    else
        inputs = dlarray(f, 'CB');
    end
    out_flat = extractdata(predict(net, inputs));
    out = equiv_output(out_flat, X);
end

function f = equiv_features(X)
    % O(p)-invariant features: entries of X X'
    [n, ~, B] = size(X);
    P = reshape(pagemtimes(X, 'none', X, 'transpose'), n*n, B);
    f = P(triu(true(n)), :);
end

function U = equiv_output(out, X)
    % u = F X, equivariant: u(XR) = u(X) R
    [n, ~, B] = size(X);
    U = pagemtimes(reshape(out, n, n, B), X);
end

function P_Z = batchProjector(X, Z)
    X_T_Z = pagemtimes(X, 'transpose', Z, 'none');
    Z_T_X = pagemtimes(Z, 'transpose', X, 'none');
    sym_part = 0.5 * (X_T_Z + Z_T_X);
    P_Z = Z - pagemtimes(X, sym_part);
end

function Q = Retraction(A)
    [Q_full, R] = qr(A, 0);
    D = diag(sign(diag(R)));
    Q = Q_full * D;
end

function [X, info] = gmc_stiefel(X0, H, beta, opts)
    [n, p, C] = size(X0);
    X   = X0;
    lam = sort(eig(H));
    Uf  = @(Y) beta * reshape(sum(Y .* pagemtimes(H, Y), [1 2]), [], 1);
    gU  = @(Y) batchProjector(Y, 2*beta*pagemtimes(H, Y));

    log_stp = log(min(opts.eps_max, 1/sqrt(2*beta*(lam(end) - lam(1)))));
    T = opts.n_warmup + opts.n_iter;
    info.E_mean  = zeros(T, 1);
    info.E_trace = zeros(opts.n_iter, C);
    acc_keep = zeros(opts.n_iter, 1);

    Ux = Uf(X);
    G  = gU(X);
    for it = 1:T
        stp = exp(log_stp) * (0.8 + 0.4*rand);
        V   = batchProjector(X, randn(n, p, C));
        K0  = 0.5 * reshape(sum(V.^2, [1 2]), [], 1);
        Xp = X;  Vp = V;  Gp = G;
        for l = 1:opts.L
            Vp = Vp - 0.5*stp*Gp;
            [Xp, Vp] = stiefel_geodesic(Xp, Vp, stp);
            Xp = batchRetraction(Xp);
            Vp = batchProjector(Xp, Vp);
            Gp = gU(Xp);
            Vp = Vp - 0.5*stp*Gp;
        end
        Up   = Uf(Xp);
        K1   = 0.5 * reshape(sum(Vp.^2, [1 2]), [], 1);
        logr = (Ux + K0) - (Up + K1);
        logr(~isfinite(logr)) = -Inf;
        acc  = log(rand(C, 1)) < logr;
        X(:,:,acc) = Xp(:,:,acc);
        G(:,:,acc) = Gp(:,:,acc);
        Ux(acc)    = Up(acc);

        a_mean = mean(exp(min(logr, 0)));
        if it <= opts.n_warmup
            log_stp = min(log(opts.eps_max), log_stp + 2*(a_mean - opts.target_acc)/it^0.6);
        else
            acc_keep(it - opts.n_warmup) = a_mean;
            info.E_trace(it - opts.n_warmup, :) = Ux' / beta;
        end
        info.E_mean(it) = mean(Ux) / beta;
    end
    info.acc_rate   = mean(acc_keep);
    info.step_size  = exp(log_stp);
    info.grad_evals = T * opts.L;
end

function [X, info] = rula_stiefel(X0, H, beta, opts)
    [n, p, C] = size(X0);
    X   = X0;
    lam = sort(eig(H));
    h   = min(opts.h_max, opts.c_h / (2*beta*(lam(end) - lam(1))));
    n_rec = floor((opts.n_steps - opts.n_burn) / opts.thin);
    info.E_mean  = zeros(opts.n_steps, 1);
    info.E_trace = zeros(n_rec, C);
    r = 0;
    for k = 1:opts.n_steps
        G  = batchProjector(X, 2*beta*pagemtimes(H, X));
        xi = batchProjector(X, randn(n, p, C));
        X  = batchRetraction(X - h*G + sqrt(2*h)*xi);
        e  = reshape(sum(X .* pagemtimes(H, X), [1 2]), 1, []);
        info.E_mean(k) = mean(e);
        if k > opts.n_burn && mod(k - opts.n_burn, opts.thin) == 0
            r = r + 1;
            info.E_trace(r, :) = e;
        end
    end
    info.acc_rate   = NaN;
    info.step_size  = h;
    info.grad_evals = opts.n_steps;
end

function [Xt, Vt] = stiefel_geodesic(X, V, t)
    [~, p, C] = size(X);
    A  = pagemtimes(X, 'transpose', V, 'none');
    S  = pagemtimes(V, 'transpose', V, 'none');
    Ip = repmat(eye(p), 1, 1, C);
    E  = batch_expm(t * [A, -S; Ip, A]);
    R  = batch_expm(-t * A);
    XV = pagemtimes([X, V], E);
    Xt = pagemtimes(XV(:, 1:p, :), R);
    Vt = pagemtimes(XV(:, p+1:end, :), R);
end

function E = batch_expm(M)
    [m, ~, C] = size(M);
    nrm = max(sum(abs(M), 1), [], 2);
    nrm = nrm(isfinite(nrm));
    if isempty(nrm)
        s = 0;
    else
        s = max(0, min(64, ceil(log2(max(nrm) / 0.25))));
    end
    Ms = M / 2^s;
    I  = repmat(eye(m), 1, 1, C);
    E  = I + Ms / 12;
    for k = 11:-1:1
        E = I + pagemtimes(Ms, E) / k;
    end
    for k = 1:s
        E = pagemtimes(E, E);
    end
end

function Q = batchRetraction(A)
    [~, p, ~] = size(A);
    Q = zeros(size(A), 'like', A);
    for j = 1:p
        v = A(:, j, :);
        for pass = 1:2
            for k = 1:j-1
                v = v - Q(:, k, :) .* sum(Q(:, k, :) .* v, 1);
            end
        end
        Q(:, j, :) = v ./ sqrt(sum(v.^2, 1));
    end
end

function ref = exact_reference_Stiefel42(H, p, beta, n_u, n_phi)
    % Exact mean, variance, occupations and energy CDF of pi on V(4,2)
    if nargin < 4, n_u = 96;   end
    if nargin < 5, n_phi = 32; end
    n = size(H, 1);
    [U, Lam] = eig((H + H')/2);
    [lam, idx] = sort(diag(Lam), 'ascend');
    ref.mean = NaN;  ref.var = NaN;  ref.occ = nan(n, 1);
    ref.U    = U(:, idx);
    ref.cdf  = @(e) nan(size(e));
    if n ~= 4 || p ~= 2
        warning('exact_reference_Stiefel42: only V(4,2) is supported; reference metrics set to NaN.');
        return;
    end

    S = sum(lam);
    d = [lam(1)+lam(2)-lam(3)-lam(4); lam(1)+lam(3)-lam(2)-lam(4); lam(1)+lam(4)-lam(2)-lam(3)];
    [~, perm] = sort(abs(d), 'descend');
    dp = d(perm);
    c  = (dp(1)^2 - max(dp(2)^2, dp(3)^2)) / (2*abs(dp(1)));
    if c > 1e-12, u_cut = min(1, 50/(beta*c)); else, u_cut = 1; end

    [u,  wu ] = gauss_legendre(n_u, 0, u_cut);
    [ph, wph] = gauss_legendre(n_phi, 0, pi/2);
    [UU, PP]  = ndgrid(u, ph);
    W  = wu * wph';
    b2 = cat(3, (1 - UU).^2, UU.*(2 - UU).*cos(PP).^2, UU.*(2 - UU).*sin(PP).^2);
    q  = sqrt(dp(1)^2*b2(:,:,1) + dp(2)^2*b2(:,:,2) + dp(3)^2*b2(:,:,3));
    q  = max(q, 1e-300);
    kap = max(beta*q/2, 1e-300);

    lw = kap + log(-expm1(-2*kap)) - log(2*kap);
    wt = W .* exp(lw - max(lw(:)));
    wt = wt / sum(wt(:));

    Lk = langevin_fn(kap);
    mb = S/2 - 0.5*q.*Lk;
    vt = var_t_fn(kap);
    ref.mean = sum(wt(:) .* mb(:));
    ref.var  = sum(wt(:) .* (0.25*q(:).^2 .* vt(:) + (mb(:) - ref.mean).^2));

    Cp = zeros(3, 1);
    for j = 1:3
        tmp   = -wt .* Lk .* dp(j) .* b2(:,:,j) ./ q;
        Cp(j) = sum(tmp(:));
    end
    Ck = zeros(3, 1);
    Ck(perm) = Cp;
    ref.occ = 0.5 + 0.5*[ Ck(1) + Ck(2) + Ck(3);
                          Ck(1) - Ck(2) - Ck(3);
                         -Ck(1) + Ck(2) - Ck(3);
                         -Ck(1) - Ck(2) + Ck(3)];
    qv = q(:)';  kv = kap(:)';  wv = wt(:);
    ref.cdf = @(e) reference_energy_cdf(e, S, qv, kv, wv);
end

function F = reference_energy_cdf(e, S, q, kap, wt)
    sz = size(e);
    e  = e(:);
    F  = zeros(numel(e), 1);
    den = -expm1(-2*kap);
    blk = 256;
    for i0 = 1:blk:numel(e)
        ii  = i0:min(i0 + blk - 1, numel(e));
        tau = min(max((2*e(ii) - S) ./ q, -1), 1);
        F(ii) = (-expm1(-kap .* (1 + tau)) ./ den) * wt;
    end
    F = reshape(F, sz);
end

function L = langevin_fn(k)
    L = 1./tanh(k) - 1./k;
    s = k < 1e-3;
    L(s) = k(s)/3 - k(s).^3/45;
end

function v = var_t_fn(k)
    v = 1./k.^2 - 1./sinh(k).^2;
    s = k < 1e-3;
    v(s) = 1/3 - k(s).^2/15;
end

function [x, w] = gauss_legendre(m, a, b)
    k  = (1:m-1)';
    bk = k ./ sqrt(4*k.^2 - 1);
    [V, D] = eig(diag(bk, 1) + diag(bk, -1));
    [x, i] = sort(diag(D));
    w = 2 * V(1, i)'.^2;
    x = (b - a)/2 * x + (a + b)/2;
    w = (b - a)/2 * w;
end

function m = empty_metrics()
    m = struct('E_hat', NaN, 'E_se', NaN, 'abs_bias', NaN, 'bias_z', NaN, 'KS', NaN, ...
        'occ_TV', NaN, 'orth_err', NaN, 'rhat', NaN, 'ess', NaN, 'acc_rate', NaN, ...
        'step_size', NaN, 'evals_per_sample', NaN, 'time_s', NaN, 'train_time_s', NaN, ...
        'orbit_kuiper', NaN, 'det_imbalance', NaN);
end

function m = sample_metrics(X, H, ref)
    [~, p, N] = size(X);
    m = empty_metrics();
    e = reshape(sum(X .* pagemtimes(H, X), [1 2]), [], 1);
    m.E_hat    = mean(e);
    m.E_se     = std(e) / sqrt(N);
    m.abs_bias = abs(m.E_hat - ref.mean);
    m.bias_z   = m.abs_bias / sqrt(ref.var / N);
    F    = ref.cdf(sort(e));
    m.KS = max(max((1:N)'/N - F), max(F - (0:N-1)'/N));
    Y   = pagemtimes(ref.U, 'transpose', X, 'none');
    occ = mean(sum(Y.^2, 2), 3);
    m.occ_TV = 0.5 * sum(abs(occ - ref.occ)) / p;
    XtX = pagemtimes(X, 'transpose', X, 'none') - eye(p);
    m.orth_err = max(sqrt(sum(XtX.^2, [1 2])), [], 'all');
    % Symmetry check (p = 2): the polar factor of U_low' X must be Haar on O(2)
    if p == 2
        A = pagemtimes(ref.U(:, 1:2), 'transpose', X, 'none');
        a = reshape(A(1,1,:), [], 1);  b = reshape(A(1,2,:), [], 1);
        c = reshape(A(2,1,:), [], 1);  d = reshape(A(2,2,:), [], 1);
        pos = (a.*d - b.*c) > 0;
        ang = atan2(c - b, a + d);                                  % rotation component
        ang(~pos) = atan2(b(~pos) + c(~pos), a(~pos) - d(~pos));    % reflection component
        us = sort(mod(ang, 2*pi) / (2*pi));
        m.orbit_kuiper  = max((1:N)'/N - us) + max(us - (0:N-1)'/N);
        m.det_imbalance = abs(mean(pos) - 0.5);
    end
end

function [rhat, ess] = ess_rhat(Y)
    T2 = floor(size(Y, 1) / 2);
    Y  = [Y(1:T2, :), Y(T2+1:2*T2, :)];
    [N, M] = size(Y);
    mu = mean(Y, 1);
    W  = mean(var(Y, 0, 1));
    var_plus = (N - 1)/N * W + var(mu);
    rhat = sqrt(var_plus / W);
    nfft = 2^nextpow2(2*N);
    F    = fft(Y - mu, nfft, 1);
    acov = real(ifft(F .* conj(F), [], 1));
    acov = mean(acov(1:N, :), 2) / N;
    rho  = 1 - (W - acov) / var_plus;
    rho(1) = 1;
    P = rho(1:2:end-1) + rho(2:2:end);
    k = find(P < 0, 1);
    if ~isempty(k), P = P(1:k-1); end
    for i = 2:numel(P)
        P(i) = min(P(i), P(i-1));
    end
    tau = max(-1 + 2*sum(P), 1/log10(N*M));
    ess = N*M / tau;
end

function format_metric_axes(ax, mode)
    if strcmp(mode, 'loglog'), set(ax, 'XScale', 'log'); end
    set(ax, 'YScale', 'log', 'XMinorTick', 'on', 'YMinorTick', 'on', 'LineWidth', 1.0, ...
        'FontSize', 13, 'FontName', 'Times New Roman', 'Box', 'on', 'GridAlpha', 0.2);
    grid(ax, 'on');
end
