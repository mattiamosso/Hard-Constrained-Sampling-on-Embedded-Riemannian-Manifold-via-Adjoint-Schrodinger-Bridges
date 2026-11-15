% R-ASBS (Alg. 1) on the earthquake vMF-KDE target on S^2, compared with Riemannian
% ULA/MALA and Riemannian flow matching against exact samples of the target.
% ("ASBS-M" in the code = R-ASBS / Extended R-ASBS in the paper)

clear; close all; clc;

%% 0. Benchmark settings
cfg.seed       = 2026;
cfg.split_mode = 'random';           % train/test split: 'random' | 'ordered'
cfg.N          = 700;                % samples per method per repeat
cfg.R          = 5;                  % independent repeats
cfg.alpha      = 0.05;
cfg.mmd_bw     = [0.05, 0.15, 0.5];  % MMD kernel widths (rad)
cfg.cells      = [10, 20];           % equal-area cells (z-bands x sectors)
cfg.cov_k      = 5;
cfg.N_ref_E    = 5000;               % exact samples for the references
cfg.N_ref_mmd  = 2000;
cfg.N_ref_mass = 200000;

% Riemannian Langevin baselines (Cheng, Zhang & Sra, 2022)
mcmc.delta_ula   = 5e-4;             % ULA step
mcmc.delta0_mala = 5e-4;             % MALA initial step (adapted)
mcmc.target_acc  = 0.574;
mcmc.budget_mult = [1/16, 1/4, 1, 4];% grad-E budget relative to ASBS-M training
mcmc.thin        = 10;

% Riemannian flow matching (Chen & Lipman, 2024); empty fields matched to ASBS-M
fm.num_freq = 256;  fm.rff_bw = 3.0;  fm.lr = 1e-3;
fm.iters = [];  fm.batch = [];  fm.N_steps = [];

%% 1. Data
fprintf('Start Import data...\n');
data = readtable('query.csv');

% Spherical to Cartesian
lat = deg2rad(data.latitude);
lon = deg2rad(data.longitude);

x_dir = cos(lat) .* cos(lon);
y_dir = cos(lat) .* sin(lon);
z_dir = sin(lat);
M_impact = [x_dir'; y_dir'; z_dir'];
valid_count = size(M_impact, 2);

% Train/test split ('ordered' = temporal split)
split_ratio = 0.7;
num_train = floor(valid_count * split_ratio);
rng(cfg.seed, 'twister');
switch cfg.split_mode
    case 'random',  order = randperm(valid_count);
    case 'ordered', order = 1:valid_count;
    otherwise, error('cfg.split_mode must be ''random'' or ''ordered''.');
end
M_impact = M_impact(:, order);
M_train = M_impact(:, 1:num_train);
M_test = M_impact(:, num_train+1:end);
fprintf('Import data completed. Valid points: %d (Train: %d, Test: %d, split: %s)\n', ...
    valid_count, size(M_train, 2), size(M_test, 2), cfg.split_mode);

%% 2. ASBS-M training
t_start = tic;
[uNet, hNet, params] = ASBS_sphere_sampler(M_train);
t_train_asbs = toc(t_start);
fprintf('\nTraining completed (%.1f s).\n', t_train_asbs);

kappa = params.kappa_end;            % target of every energy-based sampler
N_samples = cfg.N;
rng(cfg.seed + 1, 'twister');        % restart a known RNG stream
if isempty(fm.iters),   fm.iters   = params.epochs;  end
if isempty(fm.batch),   fm.batch   = params.B;       end
if isempty(fm.N_steps), fm.N_steps = params.N_steps; end

% Training budget: one grad E per particle per epoch; MCMC gets the same budget
budget_asbs = params.epochs * params.B;
steps_ck    = max(1, round(mcmc.budget_mult * budget_asbs / cfg.N));
[~, i_main] = min(abs(mcmc.budget_mult - 1));   % main-comparison checkpoint

%% 3. Samples: R repeats x N per method
method_names = {'Exact (oracle)', 'Train resample', 'ASBS-M', 'Riem. ULA', 'Riem. MALA', 'Riem. FM'};
iEx = 1; iTr = 2; iA = 3; iU = 4; iM = 5; iF = 6;
nMeth = numel(method_names);
S = cell(nMeth, cfg.R);                       % S{method, repeat} = 3 x N
cost = struct('grad_evals', nan(1, nMeth), 'net_evals_per_sample', nan(1, nMeth), ...
    'train_time', nan(1, nMeth), 'sample_time', nan(1, nMeth));

% References: exact draws and resampled training events
for r = 1:cfg.R
    S{iEx, r} = sample_kde_exact(M_train, kappa, cfg.N);
    S{iTr, r} = M_train(:, randperm(num_train, min(cfg.N, num_train)));
end
cost.grad_evals([iEx, iTr]) = 0;
cost.net_evals_per_sample([iEx, iTr, iU, iM]) = 0;
cost.train_time([iEx, iTr, iU, iM]) = 0;
cost.sample_time([iEx, iTr]) = 0;

% ASBS-M
fprintf('Generating final manifold samples (ASBS-M)...\n');
t_start = tic;
for r = 1:cfg.R
    [~, S{iA, r}] = simulate_trajectories_fixed(uNet, params, cfg.N);
end
cost.sample_time(iA) = toc(t_start) / cfg.R;
cost.train_time(iA)  = t_train_asbs;
cost.grad_evals(iA)  = budget_asbs;               % all spent in training
cost.net_evals_per_sample(iA) = params.N_steps;
M_generated = S{iA, 1};

% Riemannian ULA / MALA: R*N parallel chains, states stored at each budget checkpoint
fprintf('Running Riemannian Langevin baselines (%d chains, up to %d steps)...\n', ...
    cfg.R*cfg.N, max(steps_ck));
X0_mcmc = randn(3, cfg.R*cfg.N);
X0_mcmc = X0_mcmc ./ vecnorm(X0_mcmc, 2, 1);
[ck_ula,  info_ula ] = riemannian_langevin_sphere(X0_mcmc, M_train, kappa, mcmc.delta_ula,   false, steps_ck, mcmc);
[ck_mala, info_mala] = riemannian_langevin_sphere(X0_mcmc, M_train, kappa, mcmc.delta0_mala, true,  steps_ck, mcmc);
for r = 1:cfg.R
    cols = (r-1)*cfg.N + (1:cfg.N);
    S{iU, r} = ck_ula{i_main}(:, cols);
    S{iM, r} = ck_mala{i_main}(:, cols);
end
cost.grad_evals([iU, iM])  = steps_ck(i_main) * cfg.N;
cost.sample_time([iU, iM]) = [info_ula.time_ck(i_main), info_mala.time_ck(i_main)] / cfg.R;

% Riemannian flow matching (trained on data, not on the energy)
fprintf('Training Riemannian flow matching baseline...\n');
t_start = tic;
[vNet, fm] = train_riemannian_fm(M_train, fm);
cost.train_time(iF) = toc(t_start);
t_start = tic;
for r = 1:cfg.R
    S{iF, r} = sample_riemannian_fm(vNet, fm, cfg.N);
end
cost.sample_time(iF) = toc(t_start) / cfg.R;
cost.grad_evals(iF)  = 0;
cost.net_evals_per_sample(iF) = fm.N_steps;

%% 4. References
fprintf('\nComputing references...\n');
ctx.M_train = M_train;  ctx.M_test = M_test;  ctx.kappa = kappa;  ctx.cfg = cfg;
X_refE       = sample_kde_exact(M_train, kappa, cfg.N_ref_E);
ctx.E_refE   = energy_grad_sphere(X_refE, M_train, kappa);
ctx.E_pi     = mean(ctx.E_refE);
ctx.sd_pi    = std(ctx.E_refE);
ctx.X_refMMD = sample_kde_exact(M_train, kappa, cfg.N_ref_mmd);
ctx.mass_pi  = cell_hist(sample_kde_exact(M_train, kappa, cfg.N_ref_mass), cfg.cells);
ctx.mass_test = cell_hist(M_test, cfg.cells);
n_pair = min(cfg.N, size(M_test, 2));
ctx.pair_pi = cell(cfg.R, 1);  ctx.pair_test = cell(cfg.R, 1);
for r = 1:cfg.R    % same reference subsets for every method
    ctx.pair_pi{r}   = sample_kde_exact(M_train, kappa, n_pair);
    ctx.pair_test{r} = M_test(:, randperm(size(M_test, 2), n_pair));
end
E_test = energy_grad_sphere(M_test, M_train, kappa);

% Metrics: 'pi' = fidelity to the target, 'test' = generalisation
bw_deg = round(rad2deg(cfg.mmd_bw));
metric_names = [{'E bias z-score', 'KS energy'}, ...
    arrayfun(@(d) sprintf('MMD2 vs pi (%d deg)', d), bw_deg, 'UniformOutput', false), ...
    {'W1 vs pi (deg)', 'Cell-mass TV vs pi'}, ...
    arrayfun(@(d) sprintf('MMD2 vs test (%d deg)', d), bw_deg, 'UniformOutput', false), ...
    {'W1 vs test (deg)', 'Cell-mass TV vs test', 'Coverage of test', 'NN dist. to train (deg)'}];
nb = numel(cfg.mmd_bw);
metric_group = [repmat({'pi'}, 1, 4 + nb), repmat({'test'}, 1, 3 + nb), {'diag'}];
higher_better = false(1, numel(metric_names));
higher_better(strcmp(metric_names, 'Coverage of test')) = true;
ctx.n_metrics = numel(metric_names);

%% 5. Evaluation
fprintf('Evaluating...\n');
MET = nan(nMeth, cfg.R, ctx.n_metrics);
for m = 1:nMeth
    for r = 1:cfg.R
        MET(m, r, :) = eval_metrics(S{m, r}, ctx, r);
    end
end

% Budget sweep of the MCMC baselines
SWEEP = nan(2, numel(steps_ck), cfg.R, ctx.n_metrics);
cks = {ck_ula, ck_mala};
for s = 1:2
    for c = 1:numel(steps_ck)
        for r = 1:cfg.R
            SWEEP(s, c, r, :) = eval_metrics(cks{s}{c}(:, (r-1)*cfg.N + (1:cfg.N)), ctx, r);
        end
    end
end

% MCMC diagnostics on the second half of the main run
rows = info_ula.trace_step > steps_ck(i_main)/2 & info_ula.trace_step <= steps_ck(i_main);
if sum(rows) < 4, rows = info_ula.trace_step <= max(steps_ck); end
[rhat_ula,  ess_ula ] = ess_rhat(info_ula.E_trace(rows, :));
[rhat_mala, ess_mala] = ess_rhat(info_mala.E_trace(rows, :));

%% 6. Tables, paired tests, cost
mu  = squeeze(mean(MET, 2));
sdv = squeeze(std(MET, 0, 2));
fprintf('\n================ QUANTITATIVE EVALUATION (mean +- sd over R = %d repeats, N = %d) ================\n', cfg.R, cfg.N);
fprintf('Target pi: E_pi = %.4f (sd %.3f). Test-set mean energy = %.4f (target-vs-test gap %.4f)\n', ...
    ctx.E_pi, ctx.sd_pi, mean(E_test), mean(E_test) - ctx.E_pi);
fprintf('%-24s', 'Metric');
fprintf(' | %-19s', method_names{:});
fprintf('\n%s\n', repmat('-', 1, 24 + 22*nMeth));
for j = 1:ctx.n_metrics
    fprintf('%-24s', metric_names{j});
    for m = 1:nMeth
        fprintf(' | %9.4g +- %-7.2g', mu(m, j), sdv(m, j));
    end
    fprintf('\n');
end
fprintf(['Lower is better except coverage. Exact (oracle) = noise floor of a perfect sampler of pi;\n' ...
    'Train resample = memorisation reference (lowest possible NN distance to train).\n']);

% Paired t-tests vs ASBS-M (pi metrics: energy-based samplers only)
tcrit = t_critical(cfg.R - 1, cfg.alpha);
competitors = [iU, iM, iF];
verdict = cell(ctx.n_metrics, numel(competitors));
tstat   = nan(ctx.n_metrics, numel(competitors));
fprintf('\nPaired comparison vs ASBS-M (two-sided t-test, alpha = %.2f, t_crit = %.2f):\n', cfg.alpha, tcrit);
fprintf('%-24s', 'Metric');
fprintf(' | vs %-20s', method_names{competitors});
fprintf('\n');
for j = 1:ctx.n_metrics
    if strcmp(metric_group{j}, 'diag'), continue; end
    fprintf('%-24s', metric_names{j});
    for c = 1:numel(competitors)
        b = competitors(c);
        if strcmp(metric_group{j}, 'pi') && b == iF
            verdict{j, c} = 'n/a';
        else
            d = squeeze(MET(b, :, j) - MET(iA, :, j));      % > 0: ASBS-M lower
            if higher_better(j), d = -d; end
            tstat(j, c) = mean(d) / (std(d) / sqrt(cfg.R));
            if tstat(j, c) > tcrit,      verdict{j, c} = 'ASBS better';
            elseif tstat(j, c) < -tcrit, verdict{j, c} = 'ASBS worse';
            else,                        verdict{j, c} = 'tie';
            end
        end
        if isnan(tstat(j, c)), fprintf(' | %-23s', verdict{j, c});
        else, fprintf(' | %-12s (t=%6.2f)', verdict{j, c}, tstat(j, c)); end
    end
    fprintf('\n');
end
scored = verdict(~cellfun(@isempty, verdict) & ~strcmp(verdict, 'n/a'));
fprintf('Summary: ASBS-M better in %d, tie in %d, worse in %d of %d scored comparisons.\n', ...
    sum(strcmp(scored, 'ASBS better')), sum(strcmp(scored, 'tie')), sum(strcmp(scored, 'ASBS worse')), numel(scored));

fprintf('\nCost per set of %d samples (energy budget matched to ASBS-M training = %d grad-E evaluations):\n', cfg.N, budget_asbs);
fprintf('%-16s | %12s | %14s | %10s | %11s\n', 'Method', 'grad-E evals', 'net evals/samp', 'train [s]', 'sample [s]');
for m = 1:nMeth
    fprintf('%-16s | %12.0f | %14.0f | %10.1f | %11.2f\n', method_names{m}, cost.grad_evals(m), ...
        cost.net_evals_per_sample(m), cost.train_time(m), cost.sample_time(m));
end
fprintf('MCMC diagnostics (2nd half of the main run): ULA R-hat = %.3f, ESS = %.0f | MALA R-hat = %.3f, ESS = %.0f, acceptance = %.2f, step = %.1e\n', ...
    rhat_ula, ess_ula, rhat_mala, ess_mala, info_mala.acc_ck(i_main), info_mala.delta_ck(i_main));
fprintf('==========================================================================================\n');

% Save results
[mm, rr, jj] = ndgrid(1:nMeth, 1:cfg.R, 1:ctx.n_metrics);
T_long = table(method_names(mm(:))', rr(:), metric_names(jj(:))', MET(:), ...
    'VariableNames', {'method', 'repeat', 'metric', 'value'});
writetable(T_long, 'earthquake_benchmark_metrics.csv');
save('earthquake_benchmark_results.mat', 'cfg', 'mcmc', 'fm', 'method_names', 'metric_names', ...
    'metric_group', 'MET', 'SWEEP', 'steps_ck', 'budget_asbs', 'cost', 'verdict', 'tstat', ...
    'rhat_ula', 'rhat_mala', 'ess_ula', 'ess_mala', 'S', 'ctx');
%% 7. Figures
max_test_particles = N_samples;
test_end_idx = min(valid_count, num_train + max_test_particles);
M_test_plot = M_impact(:, num_train+1:test_end_idx);

max_train_plot = 1000;
num_train_plot = min(max_train_plot, size(M_train, 2));
idx_train_plot = round(linspace(1, size(M_train, 2), num_train_plot));
M_train_plot = M_train(:, idx_train_plot);

[sx, sy, sz] = sphere(500);

% Training data
fig1 = figure('Color', 'w', 'Position', [100 100 800 700],'Renderer', 'opengl');
surf(sx, sy, sz, 'EdgeColor', 'none', 'FaceColor', [0.94 0.94 0.94], ...
    'FaceAlpha', 0.6, 'FaceLighting', 'gouraud', 'AmbientStrength', 0.5, ...
    'DiffuseStrength', 0.7, 'SpecularStrength', 0.05);
hold on;

load coastlines;
r = 1.0;
x_land = r * cosd(coastlat) .* cosd(coastlon);
y_land = r * cosd(coastlat) .* sind(coastlon);
z_land = r * sind(coastlat);
plot3(x_land, y_land, z_land, 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);

theta = linspace(0, 2*pi, 150);
plot3(cos(theta), sin(theta), zeros(size(theta)), ...
    '--', 'LineWidth', 1.2, 'Color', [0.55 0.55 0.55]);

hTrain = scatter3(M_train_plot(1,:), M_train_plot(2,:), M_train_plot(3,:), ...
    14, 'k', 'o', 'filled', 'MarkerEdgeColor', [1 1 1], 'LineWidth', 0.4, ...
    'MarkerFaceAlpha', 0.9, 'DisplayName', 'Train Data (Sampled)');

axis equal; axis tight;
xlim([-1.1 1.1]); ylim([-1.1 1.1]); zlim([-1.1 1.1]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 24);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 24);
zlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 24);
set(gca, 'FontSize', 16, 'LineWidth', 1.0, 'FontName', 'Times New Roman', ...
    'TickLabelInterpreter', 'latex', 'Box', 'off', 'GridAlpha', 0.25, ...
    'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
grid on;
camlight('headlight'); camlight('left'); lighting gouraud; view(35, 30);
legend(hTrain, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 20, 'Box', 'off');
title('Earthquake Training Data Distribution', 'Interpreter', 'latex', 'FontSize', 24);

% Energy landscape, test data and samples
grid_pts = [sx(:)'; sy(:)'; sz(:)'];
E_vec = energy_grad_sphere(grid_pts, M_train, params.kappa_end)';   % chunked (memory)
E_matrix = reshape(E_vec, size(sx));

fig2 = figure('Color', 'w', 'Position', [150 150 800 700], 'Renderer', 'opengl');
surf(sx, sy, sz, E_matrix, 'EdgeColor', 'none', 'FaceAlpha', 0.75, ...
    'FaceLighting', 'gouraud', 'AmbientStrength', 0.6, ...
    'DiffuseStrength', 0.7, 'SpecularStrength', 0.05);
hold on;

colormap('cool');
cb = colorbar;
cb.TickLabelInterpreter = 'latex';
cb.Label.String = 'Energy $E(x)$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 24;
cb.Color = [0.15 0.15 0.15];
cb.Box = 'off';

plot3(cos(theta), sin(theta), zeros(size(theta)), ...
    '--', 'LineWidth', 1.2, 'Color', [0.55 0.55 0.55]);

hTest = scatter3(M_test_plot(1,:), M_test_plot(2,:), M_test_plot(3,:), 20, ...
    'k', 'o', 'filled', 'LineWidth', 0.4, 'MarkerFaceAlpha', 0.9, ...
    'DisplayName', 'True Data (Test Set)');

hSamples = scatter3(M_generated(1,:), M_generated(2,:), M_generated(3,:), ...
    24, [0.85, 0.35, 0.0], '^', 'filled', 'MarkerEdgeColor', [1, 1, 1], ...
    'LineWidth', 0.4, 'MarkerFaceAlpha', 0.9, 'DisplayName', 'ASBS-M Samples');

axis equal; axis tight;
xlim([-1.1 1.1]); ylim([-1.1 1.1]); zlim([-1.1 1.1]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 24);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 24);
zlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 24);
set(gca, 'FontSize', 16, 'LineWidth', 1.0, 'FontName', 'Times New Roman', ...
    'TickLabelInterpreter', 'latex', 'Box', 'off', 'GridAlpha', 0.25, ...
    'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
grid on;
camlight('headlight'); camlight('left'); lighting gouraud; view(35, 30);
legend([hTest, hSamples], 'Location', 'northeast', 'Interpreter', 'latex', ...
       'FontSize', 22, 'Box', 'off');
title('Energy Landscape and Generated Samples', 'Interpreter', 'latex', 'FontSize', 24);

%% 8. Benchmark figures
col_m = {[0.20 0.20 0.20], [0.55 0.55 0.55], [217 89 0]/255, [51 102 204]/255, [0 160 122]/255, [142 68 173]/255};
mk_m  = {'.', '.', '^', 'o', 's', 'd'};
comp_show = [iA, iU, iM, iF];

% Samples on a Hammer equal-area map
clon = coastlon;  clat = coastlat;
clon([false; abs(diff(clon)) > 180]) = NaN;          % break lines at the dateline
[cx, cy] = hammer_xy(deg2rad(clat), deg2rad(clon));
[bx1, by1] = hammer_xy(linspace(-pi/2, pi/2, 100), pi*ones(1, 100));
panels = {M_test, S{iEx, 1}, S{iA, 1}, S{iU, 1}, S{iM, 1}, S{iF, 1}};
pan_ttl = {'Test set', method_names{iEx}, method_names{iA}, method_names{iU}, method_names{iM}, method_names{iF}};
pan_col = {[0 0 0], col_m{iEx}, col_m{iA}, col_m{iU}, col_m{iM}, col_m{iF}};
fig3 = figure('Color', 'w', 'Position', [100 100 1500 650]);
tl3 = tiledlayout(fig3, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(panels)
    ax = nexttile(tl3); hold(ax, 'on');
    plot(ax, cx, cy, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
    plot(ax, [bx1, -fliplr(bx1)], [by1, fliplr(by1)], 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);
    Xk = panels{k};
    [px, py] = hammer_xy(asin(max(min(Xk(3, :), 1), -1)), atan2(Xk(2, :), Xk(1, :)));
    scatter(ax, px, py, 7, pan_col{k}, 'filled', 'MarkerFaceAlpha', 0.7);
    axis(ax, 'equal');  axis(ax, 'off');
    title(ax, sprintf('%s (n = %d)', pan_ttl{k}, size(Xk, 2)), 'Interpreter', 'none', ...
        'FontName', 'Times New Roman', 'FontSize', 16);
end

% Metrics (mean +- sd over repeats)
show = [find(strcmp(metric_names, 'E bias z-score')), find(strcmp(metric_names, 'KS energy')), ...
        find(strncmp(metric_names, 'MMD2 vs pi', 10), 1), find(strcmp(metric_names, 'W1 vs pi (deg)')), ...
        find(strcmp(metric_names, 'Cell-mass TV vs pi')), find(strncmp(metric_names, 'MMD2 vs test', 12), 1), ...
        find(strcmp(metric_names, 'W1 vs test (deg)')), find(strcmp(metric_names, 'Coverage of test'))];
fig4 = figure('Color', 'w', 'Position', [100 100 1500 650]);
tl4 = tiledlayout(fig4, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for q = 1:numel(show)
    j = show(q);
    ax = nexttile(tl4); hold(ax, 'on');
    for c = 1:numel(comp_show)
        m = comp_show(c);
        if strcmp(metric_group{j}, 'pi') && m == iF, continue; end   % FM does not target pi
        bar(ax, c, mu(m, j), 0.65, 'FaceColor', col_m{m}, 'EdgeColor', 'none');
        errorbar(ax, c, mu(m, j), sdv(m, j), 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 8);
    end
    hEx = plot(ax, [0.4, numel(comp_show) + 0.6], mu(iEx, j)*[1 1], '--', 'Color', col_m{iEx}, 'LineWidth', 1.5);
    hTr = plot(ax, [0.4, numel(comp_show) + 0.6], mu(iTr, j)*[1 1], ':',  'Color', col_m{iTr}, 'LineWidth', 1.8);
    set(ax, 'XTick', 1:numel(comp_show), 'XTickLabel', method_names(comp_show), 'XLim', [0.4, numel(comp_show) + 0.6], ...
        'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on', 'TickLabelInterpreter', 'none');
    grid(ax, 'on');
    ttl = metric_names{j};
    if higher_better(j), ttl = [ttl ' (higher better)']; else, ttl = [ttl ' (lower better)']; end
    title(ax, ttl, 'Interpreter', 'none', 'FontSize', 13);
    if q == 1
        legend(ax, [hEx, hTr], {'Exact sampler (floor)', 'Train resample'}, 'Location', 'northwest', 'Box', 'off');
    end
end

% MCMC budget sweep vs ASBS-M
sw_show = [find(strncmp(metric_names, 'MMD2 vs pi', 10), 1), find(strcmp(metric_names, 'Cell-mass TV vs pi')), ...
           find(strcmp(metric_names, 'E bias z-score'))];
evals = steps_ck * cfg.N;
fig5 = figure('Color', 'w', 'Position', [100 100 1500 430]);
tl5 = tiledlayout(fig5, 1, numel(sw_show), 'TileSpacing', 'compact', 'Padding', 'compact');
for q = 1:numel(sw_show)
    j = sw_show(q);
    ax = nexttile(tl5); hold(ax, 'on');
    h = [];
    for s = 1:2
        vals = squeeze(SWEEP(s, :, :, j));
        if size(vals, 1) ~= numel(steps_ck), vals = vals'; end
        h(end+1) = errorbar(ax, evals, mean(vals, 2), std(vals, 0, 2), ['-' mk_m{iU + s - 1}], ...
            'Color', col_m{iU + s - 1}, 'MarkerFaceColor', col_m{iU + s - 1}, 'LineWidth', 1.5); %#ok<SAGROW>
    end
    xl = [min(evals)/1.5, max(evals)*1.5];
    h(end+1) = plot(ax, xl, mu(iA, j)*[1 1], '-', 'Color', col_m{iA}, 'LineWidth', 2.5);
    col_light = col_m{iA} + 0.6*(1 - col_m{iA});
    plot(ax, xl, (mu(iA, j) + sdv(iA, j))*[1 1], '-', 'Color', col_light, 'LineWidth', 1);
    plot(ax, xl, (mu(iA, j) - sdv(iA, j))*[1 1], '-', 'Color', col_light, 'LineWidth', 1);
    h(end+1) = plot(ax, xl, mu(iEx, j)*[1 1], '--', 'Color', col_m{iEx}, 'LineWidth', 1.5);
    set(ax, 'XScale', 'log', 'XLim', xl, 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on');
    if strcmp(metric_names{j}, 'E bias z-score'), set(ax, 'YScale', 'log'); end
    yl = ylim(ax);
    h(end+1) = plot(ax, budget_asbs*[1 1], yl, ':', 'Color', 'k', 'LineWidth', 1.2);
    ylim(ax, yl);
    grid(ax, 'on');
    xlabel(ax, sprintf('Energy-gradient evaluations per %d samples', cfg.N), 'FontSize', 13);
    title(ax, metric_names{j}, 'Interpreter', 'none', 'FontSize', 13);
    if q == 1
        legend(ax, h, {method_names{iU}, method_names{iM}, 'ASBS-M (+- sd)', 'Exact sampler (floor)', ...
            'ASBS-M training budget'}, 'Location', 'northeast', 'Box', 'off', 'Interpreter', 'none');
    end
end

%% ASBS-M (Algorithm 1)

function E_map = calc_E_vectorized(X_batch, M_train, kappa)
    dots = M_train' * X_batch;
    max_dot = max(dots, [], 1);
    E_map = -(log(mean(exp(kappa * (dots - max_dot)), 1)) + kappa .* max_dot)';
end

function gradE_M = calc_gradE_M_vectorized(X_batch, M_train, kappa)
    dots = M_train' * X_batch;
    max_dot = max(dots, [], 1);
    weights = exp(kappa * (dots - max_dot));
    sum_weights = sum(weights, 1);
    grad_euclidean = - (M_train * (kappa * weights)) ./ sum_weights;
    gradE_M = grad_euclidean - sum(grad_euclidean .* X_batch, 1) .* X_batch;
end

function rff_feat = apply_rff(x, B)
    proj = 2 * pi * B * x;
    rff_feat = [cos(proj); sin(proj)];
end

function [uNet, hNet, params] = ASBS_sphere_sampler(M_train)
    rng(1, 'twister');

    params.B = 500;
    params.epochs = 2000;
    params.N_steps = 250;
    params.dt = 1.0 / params.N_steps;
    params.max_drift = 30.0;

    % Geometric sigma(t) = sig_max (sig_min/sig_max)^t
    % Optional Ricci damping of the adjoint (false = Algorithm 1 as printed)
    params.ricci_damping = false;
    if params.ricci_damping
        params.sig_max = 3.0;          % V1 = 0.98
    else
        params.sig_max = 2.0;          % V1 = 0.48
    end
    params.sig_min = 0.03;
    L_sig = log(params.sig_max / params.sig_min);
    params.sigma_fn = @(t) params.sig_max * exp(-L_sig * t);

    % V(t) = int_0^t sigma_s^2 ds
    params.V_actual = @(t) params.sig_max^2 * (1 - exp(-2*L_sig*t)) / (2*L_sig);
    params.V1 = params.V_actual(1.0);

    % n_inner Adam steps per epoch (fresh bridge points, no extra energy evaluations)
    params.n_inner = 20;

    params.kappa_end = 600;

    % Random Fourier features
    num_freq = 256;
    rff_bandwidth = 3.0;
    params.B_u = randn(num_freq, 4) * rff_bandwidth;
    params.B_u(:, 4) = randn(num_freq, 1) * 1.0;
    params.B_h = randn(num_freq, 3) * rff_bandwidth;

    % Learning rates
    lr_u_init = 1e-3;
    lr_h_init = 2e-3;
    lr_end = 0.1;               % cosine decay to 10%

    % Networks
    layers_u = [
        featureInputLayer(num_freq * 2, 'Name', 'input_u')
        fullyConnectedLayer(256,'Name', 'fc1_u')
        layerNormalizationLayer('Name', 'ln1_u')
        geluLayer('Name', 'gelu1_u')
        fullyConnectedLayer(256,'Name', 'fc2_u')
        layerNormalizationLayer('Name', 'ln2_u')
        geluLayer('Name', 'gelu2_u')
        fullyConnectedLayer(256,'Name', 'fc3_u')
        layerNormalizationLayer('Name', 'ln3_u')
        geluLayer('Name', 'gelu3_u')
        fullyConnectedLayer(3, 'Name', 'out_u')
    ];
    uNet = dlnetwork(layers_u);
    uNet = initialize(uNet);

    layers_h = [
        featureInputLayer(num_freq * 2, 'Name', 'input_h')
        fullyConnectedLayer(256,'Name', 'fc1_h')
        layerNormalizationLayer('Name', 'ln1_h')
        geluLayer('Name', 'gelu1_h')
        fullyConnectedLayer(256,'Name', 'fc2_h')
        layerNormalizationLayer('Name', 'ln2_h')
        geluLayer('Name', 'gelu2_h')
        fullyConnectedLayer(256,'Name', 'fc3_h')
        layerNormalizationLayer('Name', 'ln3_h')
        geluLayer('Name', 'gelu3_h')
        fullyConnectedLayer(3, 'Name', 'out_h')
    ];
    hNet = dlnetwork(layers_h);
    hNet = initialize(hNet);

    u_avgG = []; u_avgSqG = []; u_iter = 0;
    h_avgG = []; h_avgSqG = []; h_iter = 0;

    fprintf('Starting ASBS-M Training Loop...\n');
    for epoch = 1:params.epochs
        lr_fac = lr_end + (1 - lr_end) * 0.5 * (1 + cos(pi * (epoch - 1) / params.epochs));
        lr_u = lr_u_init * lr_fac;
        lr_h = lr_h_init * lr_fac;

        % Staircase annealing of kappa
        if epoch <= params.epochs * 0.25
            curr_kappa = 150;
        elseif epoch <= params.epochs * 0.50
            curr_kappa = 300;
        elseif epoch <= params.epochs * 0.75
            curr_kappa = 450;
        else
            curr_kappa = params.kappa_end;
        end

        % Simulate paths (drift clipping)
        [X0, X1] = simulate_trajectories_fixed(uNet, params, params.B);

        % Adjoint matching
        a_1 = terminal_adjoint(X1, hNet, params, M_train, curr_kappa);
        for it = 1:params.n_inner
            [Xt, t_vec, a_t, sigma_t_vec] = compute_adjoint_targets(X0, X1, a_1, params);

            dl_Xt_input_raw = [Xt; t_vec];
            dl_Xt_input_rff = dlarray(apply_rff(dl_Xt_input_raw, params.B_u), 'CB');
            dl_Xt_pos = dlarray(Xt, 'CB');
            dl_target_a = dlarray(a_t, 'CB');
            dl_sigma = dlarray(sigma_t_vec, 'CB');

            u_iter = u_iter + 1;
            [loss_u, grads_u] = dlfeval(@modelLossU, uNet, dl_Xt_input_rff, dl_target_a, dl_Xt_pos, dl_sigma);
            [uNet.Learnables, u_avgG, u_avgSqG] = adamupdate(uNet.Learnables, grads_u, u_avgG, u_avgSqG, u_iter, lr_u);
        end

        % Corrector matching
        [X0_new, X1_new] = simulate_trajectories_fixed(uNet, params, params.B);
        b_target = compute_corrector_targets(X0_new, X1_new, params);

        dl_X1_input_raw = X1_new;
        dl_X1_input_rff = dlarray(apply_rff(dl_X1_input_raw, params.B_h), 'CB');
        dl_target_b = dlarray(b_target, 'CB');
        dl_X1_pos = dlarray(X1_new, 'CB');

        for it = 1:params.n_inner
            h_iter = h_iter + 1;
            [loss_h, grads_h] = dlfeval(@modelLossH, hNet, dl_X1_input_rff, dl_target_b, dl_X1_pos);
            [hNet.Learnables, h_avgG, h_avgSqG] = adamupdate(hNet.Learnables, grads_h, h_avgG, h_avgSqG, h_iter, lr_h);
        end

        if mod(epoch, 50) == 0 || epoch == 1
            fprintf('Epoch %4d/%4d | Kappa: %7.2f | Ctrl Loss: %8.5f | Corr Loss: %8.5f\n', ...
                epoch, params.epochs, curr_kappa, extractdata(loss_u), extractdata(loss_h));
        end
    end
end

function [X0_dl, X1_dl] = simulate_trajectories_fixed(uNet, params, num_samples)
    X0_dl = randn(3, num_samples);
    X0_dl = X0_dl ./ vecnorm(X0_dl, 2, 1);
    X1_dl = X0_dl;

    for step = 1:params.N_steps
        t = (step - 1) * params.dt;
        sigma_t = params.sigma_fn(t);

        raw_input = [X1_dl; t * ones(1, num_samples)];
        rff_input = apply_rff(raw_input, params.B_u);
        dl_in = dlarray(rff_input, 'CB');

        u_val = extractdata(predict(uNet, dl_in));
        u_proj = u_val - sum(u_val .* X1_dl, 1) .* X1_dl;

        drift_norms = vecnorm(u_proj, 2, 1);
        scale_factor = min(1.0, params.max_drift ./ (drift_norms + 1e-8));
        u_clipped = u_proj .* scale_factor;

        dW = randn(3, num_samples) * sqrt(params.dt);
        dW_proj = dW - sum(dW .* X1_dl, 1) .* X1_dl;

        X1_dl = X1_dl + sigma_t * u_clipped * params.dt + sigma_t * dW_proj;
        X1_dl = X1_dl ./ vecnorm(X1_dl, 2, 1);
    end
end

function a_1 = terminal_adjoint(X1, hNet, params, M_train, kappa)
    % a_1 = grad_M E(X1) + P h(X1)
    rff_input_X1 = apply_rff(X1, params.B_h);
    dl_X1 = dlarray(rff_input_X1, 'CB');
    h_val = extractdata(predict(hNet, dl_X1));
    h_val = h_val - sum(h_val .* X1, 1) .* X1;

    grad_riem_E = calc_gradE_M_vectorized(X1, M_train, kappa);
    a_1 = grad_riem_E + h_val;
end

function [Xt, t_vec, a_t, sigma_t_vec] = compute_adjoint_targets(X0, X1, a_1, params)
    t_vec = rand(1, params.B);
    sigma_t_vec = params.sigma_fn(t_vec);

    dot_01 = max(min(sum(X0 .* X1, 1), 1.0), -1.0);
    theta = acos(dot_01);

    Xt = X1; a_t = a_1;
    valid_mask = theta >= 1e-5;

    if any(valid_mask)
        x0_v = X0(:, valid_mask); x1_v = X1(:, valid_mask);
        th_v = theta(valid_mask); t_v = t_vec(valid_mask);
        d01_v = dot_01(valid_mask); a1_v = a_1(:, valid_mask);

        u_dir_unnorm = x1_v - x0_v .* d01_v;
        u_dir = u_dir_unnorm ./ (vecnorm(u_dir_unnorm, 2, 1) + 1e-8);

        Vt = params.V_actual(t_v);
        V1 = params.V1;

        bridge_std = sqrt( max(0, (Vt .* (V1 - Vt)) ./ V1) );

        frac_t = Vt ./ V1;
        mu_t = x0_v .* cos(frac_t .* th_v) + u_dir .* sin(frac_t .* th_v);

        noise = randn(3, sum(valid_mask));
        tangent_noise = noise - sum(noise .* mu_t, 1) .* mu_t;

        xt_unnorm = mu_t + tangent_noise .* bridge_std;
        xt = xt_unnorm ./ vecnorm(xt_unnorm, 2, 1);
        Xt(:, valid_mask) = xt;

        dot_1t = max(min(sum(x1_v .* xt, 1), 1.0), -1.0);
        phi = acos(dot_1t);

        pt_mask = phi > 1e-5;
        if any(pt_mask)
            x1_pt = x1_v(:, pt_mask); xt_pt = xt(:, pt_mask);
            phi_pt = phi(pt_mask); d1t_pt = dot_1t(pt_mask);
            a1_pt = a1_v(:, pt_mask);

            u_pt_unnorm = xt_pt - x1_pt .* d1t_pt;
            u_pt = u_pt_unnorm ./ vecnorm(u_pt_unnorm, 2, 1);

            a1_parallel_mag = sum(a1_pt .* u_pt, 1);
            a1_parallel = u_pt .* a1_parallel_mag;
            a1_ortho = a1_pt - a1_parallel;

            a_t_pt = a1_ortho + a1_parallel .* cos(phi_pt) - (x1_pt .* a1_parallel_mag) .* sin(phi_pt);

            a_t_v = a1_v; a_t_v(:, pt_mask) = a_t_pt;
            a_t(:, valid_mask) = a_t_v;
        end
    end
    if isfield(params, 'ricci_damping') && params.ricci_damping            % optional Ricci damping
        a_t = a_t .* exp(-(params.V1 - params.V_actual(t_vec)) / 2);
    end
end

function b_target = compute_corrector_targets(X0, X1, params)
    dot_01 = max(min(sum(X1 .* X0, 1), 1.0 - 1e-6), -1.0 + 1e-6);
    theta = acos(dot_01);

    b_target = zeros(3, params.B);
    valid_mask = theta > 1e-5;

    if any(valid_mask)
        x0_v = X0(:, valid_mask);
        x1_v = X1(:, valid_mask);
        th_v = min(theta(valid_mask), pi - 0.1);

        u_rev_unnorm = x0_v - x1_v .* dot_01(valid_mask);
        u_rev = u_rev_unnorm ./ (vecnorm(u_rev_unnorm, 2, 1) + 1e-8);

        chord_term = th_v ./ params.V1;

        vol_term = zeros(size(th_v));
        small = th_v < 0.05;
        vol_term(~small) = 0.5*(cot(th_v(~small)) - 1./th_v(~small));
        vol_term(small)  = -th_v(small)/6 - (th_v(small).^3)/90;

        b_target(:, valid_mask) = u_rev .* (chord_term + vol_term);
    end
end

function [loss, gradients] = modelLossU(net, inputs_rff, targets, Xt, sigma_t_vec)
    u_pred = forward(net, inputs_rff);
    u_proj = u_pred - sum(u_pred .* Xt, 1) .* Xt;
    errors = u_proj + sigma_t_vec .* targets;

    loss = mean(sum(errors.^2, 1));
    gradients = dlgradient(loss, net.Learnables);
end

function [loss, gradients] = modelLossH(net, inputs_rff, targets, X1_pos)
    h_pred = forward(net, inputs_rff);
    h_proj = h_pred - sum(h_pred .* X1_pos, 1) .* X1_pos;
    errors = h_proj - targets;

    loss = mean(sum(errors.^2, 1));
    gradients = dlgradient(loss, net.Learnables);
end

% Exponential map on S^2
function y = sphere_expmap_batch(X, V)
    nv = vecnorm(V, 2, 1);
    zero_idx = nv < 1e-12;
    y = X + V;
    non_zero = ~zero_idx;
    if any(non_zero)
        y(:, non_zero) = X(:, non_zero) .* cos(nv(non_zero)) + ...
            (V(:, non_zero) ./ nv(non_zero)) .* sin(nv(non_zero));
    end
    y = y ./ vecnorm(y, 2, 1);
end

%% Baselines, exact reference, metrics

function X = sample_kde_exact(M_train, kappa, N)
    % Exact samples of the vMF mixture (Wood, 1994)
    mu = M_train(:, randi(size(M_train, 2), 1, N));
    u  = rand(1, N);
    w  = 1 + log(u + (1 - u) .* exp(-2*kappa)) / kappa;     % cosine of the angle to the centre
    v  = randn(3, N);
    v  = v - sum(v .* mu, 1) .* mu;
    v  = v ./ vecnorm(v, 2, 1);
    X  = mu .* w + v .* sqrt(max(0, 1 - w.^2));
end

function [E, G] = energy_grad_sphere(X, M_train, kappa)
    % Energy and Riemannian gradient, chunked over columns (E: 1 x C)
    C = size(X, 2);
    E = zeros(1, C);
    if nargout > 1, G = zeros(3, C); end
    chunk = max(1, floor(2e7 / size(M_train, 2)));
    for i0 = 1:chunk:C
        ii   = i0:min(i0 + chunk - 1, C);
        Xi   = X(:, ii);
        dots = M_train' * Xi;
        mx   = max(dots, [], 1);
        w    = exp(kappa * (dots - mx));
        sw   = sum(w, 1);
        E(ii) = -(log(sw / size(M_train, 2)) + kappa * mx);
        if nargout > 1
            g = -(M_train * (kappa * w)) ./ sw;
            G(:, ii) = g - sum(g .* Xi, 1) .* Xi;
        end
    end
end

function [X_ck, info] = riemannian_langevin_sphere(X0, M_train, kappa, delta0, use_mh, steps_ck, mcmc)
    % Riemannian ULA / MALA on S^2 (Cheng, Zhang & Sra, 2022); states stored at steps_ck.
    % MALA: step adapted towards target_acc; the exp-map Jacobian cancels.
    C = size(X0, 2);
    n_steps = max(steps_ck);
    X = X0;
    [E, G] = energy_grad_sphere(X, M_train, kappa);
    log_d = log(delta0);
    X_ck = cell(numel(steps_ck), 1);
    info.time_ck  = nan(numel(steps_ck), 1);
    info.delta_ck = nan(numel(steps_ck), 1);
    info.acc_ck   = nan(numel(steps_ck), 1);
    n_rec = floor(n_steps / mcmc.thin);
    info.E_trace    = zeros(n_rec, C);
    info.trace_step = (1:n_rec)' * mcmc.thin;
    acc_hist = nan(n_steps, 1);
    t0 = tic;
    for k = 1:n_steps
        d = exp(log_d);
        b = -0.5 * G;
        z = randn(3, C);
        z = z - sum(z .* X, 1) .* X;
        V = d * b + sqrt(d) * z;
        Y = sphere_expmap_batch(X, V);
        [Ey, Gy] = energy_grad_sphere(Y, M_train, kappa);
        if use_mh
            Vb    = sphere_logmap_batch(Y, X) - d * (-0.5 * Gy);   % reverse move
            log_a = -(Ey - E) - (sum(Vb.^2, 1) - sum((V - d*b).^2, 1)) / (2*d);
            log_a(~isfinite(log_a)) = -Inf;
            acc = log(rand(1, C)) < log_a;
            X(:, acc) = Y(:, acc);  E(acc) = Ey(acc);  G(:, acc) = Gy(:, acc);
            acc_hist(k) = mean(exp(min(log_a, 0)));
            log_d = log_d + 2 * (acc_hist(k) - mcmc.target_acc) / k^0.6;
        else
            X = Y;  E = Ey;  G = Gy;
        end
        if mod(k, mcmc.thin) == 0
            info.E_trace(k / mcmc.thin, :) = E;
        end
        for c = find(steps_ck == k)
            X_ck{c} = X;
            info.time_ck(c)  = toc(t0);
            info.delta_ck(c) = exp(log_d);
            info.acc_ck(c)   = mean(acc_hist(max(1, k - 99):k));
        end
    end
end

function V = sphere_logmap_batch(X, Y)
    % Log map on S^2
    c  = max(min(sum(X .* Y, 1), 1), -1);
    r  = acos(c);
    U  = Y - c .* X;
    V  = U .* (r ./ max(vecnorm(U, 2, 1), 1e-300));
end

function [vNet, fm] = train_riemannian_fm(M_train, fm)
    % Riemannian flow matching on S^2 (Chen & Lipman, 2024), geodesic paths
    fm.B_v = randn(fm.num_freq, 4) * fm.rff_bw;
    fm.B_v(:, 4) = randn(fm.num_freq, 1) * 1.0;
    layers_v = [
        featureInputLayer(fm.num_freq * 2, 'Name', 'input_v')
        fullyConnectedLayer(256,'Name', 'fc1_v')
        layerNormalizationLayer('Name', 'ln1_v')
        geluLayer('Name', 'gelu1_v')
        fullyConnectedLayer(256,'Name', 'fc2_v')
        layerNormalizationLayer('Name', 'ln2_v')
        geluLayer('Name', 'gelu2_v')
        fullyConnectedLayer(256,'Name', 'fc3_v')
        layerNormalizationLayer('Name', 'ln3_v')
        geluLayer('Name', 'gelu3_v')
        fullyConnectedLayer(3, 'Name', 'out_v')
    ];
    vNet = dlnetwork(layers_v);
    vNet = initialize(vNet);
    avgG = [];  avgSqG = [];
    N = size(M_train, 2);
    fm.loss = zeros(fm.iters, 1);
    for it = 1:fm.iters
        x0 = randn(3, fm.batch);
        x0 = x0 ./ vecnorm(x0, 2, 1);
        x1 = M_train(:, randi(N, 1, fm.batch));
        t  = rand(1, fm.batch);
        [xt, ut] = geodesic_path(x0, x1, t);
        dl_in = dlarray(apply_rff([xt; t], fm.B_v), 'CB');
        [loss, grads] = dlfeval(@modelLossFM, vNet, dl_in, dlarray(ut, 'CB'), dlarray(xt, 'CB'));
        [vNet.Learnables, avgG, avgSqG] = adamupdate(vNet.Learnables, grads, avgG, avgSqG, it, fm.lr);
        fm.loss(it) = double(extractdata(loss));
        if mod(it, 250) == 0 || it == 1
            fprintf('FM iter %4d/%4d | loss %.5f\n', it, fm.iters, fm.loss(it));
        end
    end
end

function [xt, ut] = geodesic_path(x0, x1, t)
    % Geodesic point and velocity at time t
    c  = max(min(sum(x0 .* x1, 1), 1), -1);
    th = acos(c);
    e  = x1 - c .* x0;
    e  = e ./ max(vecnorm(e, 2, 1), 1e-12);
    xt = x0 .* cos(t .* th) + e .* sin(t .* th);
    ut = th .* (-x0 .* sin(t .* th) + e .* cos(t .* th));
end

function [loss, gradients] = modelLossFM(net, inputs_rff, targets, Xt)
    v = forward(net, inputs_rff);
    v_proj = v - sum(v .* Xt, 1) .* Xt;
    loss = mean(sum((v_proj - targets).^2, 1));
    gradients = dlgradient(loss, net.Learnables);
end

function X = sample_riemannian_fm(vNet, fm, N)
    % Euler integration with the exponential map
    X  = randn(3, N);
    X  = X ./ vecnorm(X, 2, 1);
    dt = 1 / fm.N_steps;
    for k = 1:fm.N_steps
        t = (k - 1) * dt;
        v = extractdata(predict(vNet, dlarray(apply_rff([X; t * ones(1, N)], fm.B_v), 'CB')));
        v = v - sum(v .* X, 1) .* X;
        X = sphere_expmap_batch(X, dt * v);
    end
end

function v = eval_metrics(X, ctx, r)
    % All metrics of one sample set (order of metric_names)
    cfg = ctx.cfg;
    N = size(X, 2);
    E = energy_grad_sphere(X, ctx.M_train, ctx.kappa);
    v = zeros(1, ctx.n_metrics);
    j = 0;
    % Fidelity to pi
    j = j + 1;  v(j) = abs(mean(E) - ctx.E_pi) / (ctx.sd_pi / sqrt(N));   % |bias| / MC std. error
    j = j + 1;  v(j) = ks2(E, ctx.E_refE);
    for b = 1:numel(cfg.mmd_bw)
        j = j + 1;  v(j) = mmd2_vmf(X, ctx.X_refMMD, cfg.mmd_bw(b));
    end
    Pp = ctx.pair_pi{r};
    j = j + 1;  v(j) = w1_geodesic(X(:, 1:size(Pp, 2)), Pp);
    j = j + 1;  v(j) = 0.5 * sum(abs(cell_hist(X, cfg.cells) - ctx.mass_pi));
    % Generalisation (test set)
    for b = 1:numel(cfg.mmd_bw)
        j = j + 1;  v(j) = mmd2_vmf(X, ctx.M_test, cfg.mmd_bw(b));
    end
    Pt = ctx.pair_test{r};
    j = j + 1;  v(j) = w1_geodesic(X(:, 1:size(Pt, 2)), Pt);
    j = j + 1;  v(j) = 0.5 * sum(abs(cell_hist(X, cfg.cells) - ctx.mass_test));
    j = j + 1;  v(j) = coverage_knn(Pt, X, cfg.cov_k);
    % Mean distance to the nearest training event
    j = j + 1;  v(j) = rad2deg(mean(acos(min(max(max(ctx.M_train' * X, [], 1), -1), 1))));
end

function D = ks2(a, b)
    % Two-sample KS statistic
    a = a(:);  b = b(:);
    [vs, idx] = sort([a; b]);
    lab = [ones(numel(a), 1) / numel(a); -ones(numel(b), 1) / numel(b)];
    cs  = cumsum(lab(idx));
    D   = max(abs(cs([diff(vs) ~= 0; true])));
end

function mmd2 = mmd2_vmf(X, Y, s)
    % Unbiased MMD^2, kernel exp((x'y - 1)/s^2)
    n = size(X, 2);  m = size(Y, 2);
    Kxx = exp((X' * X - 1) / s^2);
    Kyy = exp((Y' * Y - 1) / s^2);
    Kxy = exp((X' * Y - 1) / s^2);
    mmd2 = (sum(Kxx(:)) - n) / (n*(n-1)) + (sum(Kyy(:)) - m) / (m*(m-1)) - 2 * mean(Kxy(:));
end

function w = w1_geodesic(X, Y)
    % Exact W1 (deg), geodesic cost, optimal assignment
    C = acos(max(min(X' * Y, 1), -1));
    P = matchpairs(C, 1e3);
    w = rad2deg(mean(C(sub2ind(size(C), P(:, 1), P(:, 2)))));
end

function c = coverage_knn(Ref, X, k)
    % Coverage (Naeem et al., 2020)
    Drr = acos(max(min(Ref' * Ref, 1), -1));
    Drr(1:size(Drr, 1)+1:end) = Inf;
    srt = sort(Drr, 2);
    Drx = acos(max(min(Ref' * X, 1), -1));
    c = mean(min(Drx, [], 2) < srt(:, k));
end

function h = cell_hist(X, cells)
    % Fraction of points per equal-area cell
    nb = cells(1);  ns = cells(2);
    b = max(min(floor((X(3, :) + 1) / 2 * nb), nb - 1), 0);
    s = max(min(floor((atan2(X(2, :), X(1, :)) + pi) / (2*pi) * ns), ns - 1), 0);
    h = accumarray((b * ns + s + 1)', 1, [nb * ns, 1]) / size(X, 2);
end

function [rhat, ess] = ess_rhat(Y)
    % Split R-hat and ESS (Vehtari et al., 2021)
    T2 = floor(size(Y, 1) / 2);
    if T2 < 2, rhat = NaN; ess = NaN; return; end
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

function tc = t_critical(df, alpha)
    % Two-sided Student-t critical value
    x  = betaincinv(alpha, df/2, 0.5);
    tc = sqrt(df * (1 - x) / x);
end

function [x, y] = hammer_xy(lat, lon)
    % Hammer equal-area projection
    den = sqrt(1 + cos(lat) .* cos(lon / 2));
    x = 2*sqrt(2) * cos(lat) .* sin(lon / 2) ./ den;
    y = sqrt(2) * sin(lat) ./ den;
end
