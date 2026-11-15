% Extended R-ASBS (Alg. 2) on a 10-DOF planar closed-loop chain with obstacles,
% compared with HardFlow and a constrained-MCMC reference.
% ("ASBS-M" in the code = R-ASBS / Extended R-ASBS in the paper)

clear; clc; close all;

%% 1. Configuration
beta = 10.0;
fprintf('Starting ASBS-M Training for 10-DOF Robot at beta = %.1f\n', beta);

% Robot
params.d = 10;                        % number of joints
params.L = ones(params.d, 1);         % link lengths
params.xT = 8.0;                      % target end-effector pose (x, y, theta)
params.yT = 0.0;
params.thetaT = 0.0;

% Obstacles and energy
params.obs = [3, 2; 3, 0; 3, -2;
              5, 1; 5, -1];
params.sigma_obs = 0.5;

params.alpha = 20.0;                  % repulsion scale
params.lambda = 0.05;                 % weak prior weight
params.q_rest = zeros(params.d, 1);   % rest posture

% SDE and training
params.B = 600;                       % particles
params.N_steps = 200;                 % SDE steps
params.dt = 1/params.N_steps;
params.sigma_max = 2.0;               % sigma(t) = sigma_max^(1-t) sigma_min^t
params.sigma_min = 0.1;
t_grid = (0:params.N_steps-1)' * params.dt;
params.sig = params.sigma_max.^(1 - t_grid) .* params.sigma_min.^t_grid;
params.sigma = params.sig;
params.V1 = sum(params.sig.^2) * params.dt;                               % total reference variance
params.K_epochs = 600;
params.lr = 1e-3;
params.n_inner = 20;                  % Adam steps per epoch
params.n_sub = 10;                    % minibatch = n_sub*B triples
params.src_mu = 0.1;                  % source: Retraction(src_mu + src_sd*randn)
params.src_sd = 0.2;
params.mirror = true;                 % mirror-symmetric source, odd networks
params.loss_weight = true;            % 1/sigma_t^2 loss weight (false = unweighted)

%% 2. Networks
% Controller u(q, t)
layersU = [
    featureInputLayer(params.d + 1, 'Normalization', 'none')
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(params.d)
];
netU = dlnetwork(layersU);

% Corrector h(q)
layersH = [
    featureInputLayer(params.d, 'Normalization', 'none')
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(params.d)
];
netH = dlnetwork(layersH);

[avgU, avgSqU, avgH, avgSqH] = deal([]);

%% 3. Training (Algorithm 2)
[iterU, iterH] = deal(0);
for k = 1:params.K_epochs
    % Source samples on M
    q0 = sample_source(params.B, params);

    % Forward simulation
    q_traj = simulate_trajectories(netU, q0, params);

    % Terminal adjoint v_N = P(grad E + h)
    q_N = q_traj(:,:, params.N_steps + 1);
    h_val = predict_net(netH, q_N, params.d, params.B, false, 0, params);

    [gradE, ~] = compute_energy_grad(q_N, beta, params);
    v_curr = batchProjector(q_N, gradE + h_val, params);

    % Backward adjoint transport by projection
    V_traj = zeros(params.d, params.B, params.N_steps);
    for j = params.N_steps:-1:1
        q_j = q_traj(:,:, j);
        V_traj(:,:, j) = batchProjector(q_j, v_curr, params);
        v_curr = V_traj(:,:, j);
    end

    % Controller update
    q_traj_flat = reshape(q_traj(:,:, 1:params.N_steps), params.d, params.B * params.N_steps);
    V_traj_flat = reshape(V_traj, params.d, params.B * params.N_steps);
    T_traj = kron(t_grid', ones(1, params.B));
    S_traj = kron(params.sig', ones(1, params.B));

    M_all = params.B * params.N_steps;  M_sub = min(M_all, params.n_sub * params.B);
    for it = 1:params.n_inner
        idx = randi(M_all, 1, M_sub);
        iterU = iterU + 1;
        [gradU, lossU] = dlfeval(@computeLossU, netU, q_traj_flat(:, idx), T_traj(idx), ...
            V_traj_flat(:, idx), S_traj(idx), M_sub, params);
        [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, iterU, params.lr);
    end

    % Resample with the updated controller
    q_resampled = simulate_trajectories(netU, q0, params);
    q_1 = q_resampled(:,:, params.N_steps + 1);

    % Corrector update (projected chord)
    chord = (q_1 - q0) / params.V1;
    b_target = -batchProjector(q_1, chord, params);

    for it = 1:params.n_inner
        iterH = iterH + 1;
        [gradH, lossH] = dlfeval(@computeLossH, netH, q_1, b_target, params.d, params.B, params);
        [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, iterH, params.lr);
    end

    if mod(k, 50) == 0 || k == 1
        fprintf('Epoch %04d | Loss U: %.4f | Loss H: %.4f\n', k, extractdata(lossU), extractdata(lossH));
    end
end

%% 4. Sampling and plot
fprintf('\nGenerating Multi-Modal Valid Configurations...\n');
N_test = 100;
q0_test = sample_source(N_test, params);
final_trajs = simulate_trajectories(netU, q0_test, params);
q_final = final_trajs(:,:, end);

fig = figure('Units', 'inches', 'Position', [1, 1, 8, 6], 'Color', 'w');
hold on; axis equal; box on;
title('\textbf{Multi-Modal Valid Configurations of a Closed-Loop Chain}', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$X$ Workspace', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$Y$ Workspace', 'Interpreter', 'latex', 'FontSize', 14);

% Obstacles
theta_circ = linspace(0, 2*pi, 100);
for k = 1:size(params.obs, 1)
    Ox = params.obs(k, 1); Oy = params.obs(k, 2);
    fill(Ox + params.sigma_obs*cos(theta_circ), Oy + params.sigma_obs*sin(theta_circ), ...
        [0.7 0.7 0.7], 'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 1.5, 'FaceAlpha', 0.5);
    text(Ox, Oy, sprintf('$\\mathcal{O}_%d$', k), 'Interpreter', 'latex', ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
end

% Target
plot(params.xT, params.yT, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', '#FFD700', 'LineWidth', 1.5);
text(params.xT + 0.3, params.yT, '$c(q)=0$', 'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');

% Sampled configurations
c_blue = [0, 0.4470, 0.7410];
c_red  = [0.8500, 0.3250, 0.0980];
for i = 1:N_test
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q_final(:, i), params);
    x_coords = [0; x_j]; y_coords = [0; y_j];

    % Colour by mode (above/below the obstacles)
    if y_j(5) > 1.0
        c_line = c_blue;
    else
        c_line = c_red;
    end

    plot(x_coords, y_coords, '-', 'Color', [c_line, 0.15], 'LineWidth', 2);
    plot(x_coords, y_coords, 'o', 'Color', [c_line, 0.2], 'MarkerSize', 3, 'MarkerFaceColor', 'w');
end

% Highlight two samples
highlight_idx = [2, 40];
for i = 1:length(highlight_idx)
    idx = highlight_idx(i);
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q_final(:, idx), params);
    x_coords = [0; x_j]; y_coords = [0; y_j];

    if y_j(5) > 1.0, c_line = c_blue; else, c_line = c_red; end

    plot(x_coords, y_coords, '-', 'Color', c_line, 'LineWidth', 3);
    plot(x_coords, y_coords, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
end

plot(0, 0, 'k^', 'MarkerSize', 14, 'MarkerFaceColor', '#D3D3D3', 'LineWidth', 1.5);
xlim([-1, 10]); ylim([-3, 5]);

%% 5. Evaluation and comparison with HardFlow
% Target pi propto exp(-beta E) on M = {c = 0}; by mirror symmetry each mode has mass 1/2
rng(2026, 'twister');

% Evaluation settings
ev.n_batches = 10;                 % pool = n_batches x N_test samples
ev.feas_tol  = 1e-3;               % |c|_inf tolerance
ev.r_obs     = params.sigma_obs;   % obstacle radius
ev.div_max   = 1000;               % subsample for diversity
ev.ks_min    = [20, 100];          % min class sizes for KS

% HardFlow (Li, Alim & Azizan, 2025): exact flow-matching field of the source, cost beta*E
hf.mu        = params.src_mu;  hf.s = params.src_sd;  hf.mirror = params.mirror;
hf.N_flow    = 10;                 % flow steps (paper: 10)
hf.active    = 0.5;                % optimise in the last 50% of steps
hf.lambda_oc = [0.01, 0.1, 1, 10]; % lambda_oc sweep
hf.max_iter  = 100;                % Eq. (11) iterations
hf.tol = 1e-3;  hf.eta0 = 1e-2;  hf.eta_max = 0.1;  hf.max_bt = 25;

% Reference: constrained MCMC (Zappa, Holmes-Cerfon & Goodman, 2018), within homotopy classes
mc.run = true;
mc.n_chains = 2000;  mc.n_steps = 3000;  mc.burn = 1500;  mc.thin = 100;
mc.step0 = 0.05;  mc.target_acc = 0.3;
mc.newton_iters = 30;  mc.newton_tol = 1e-10;  mc.rev_tol = 1e-7;

obs_mirror = abs(params.obs(:,1) - params.obs(:,1)') + abs(-params.obs(:,2) - params.obs(:,2)');
symmetric = all(params.q_rest == 0) && params.yT == 0 && params.thetaT == 0 && ...
    all(min(obs_mirror, [], 2) < 1e-9);         % each obstacle has a mirror image
if ~symmetric
    warning('Problem is not mirror-symmetric: mode-balance and mirror-symmetry metrics are not exact.');
end

% ASBS-M pool
fprintf('\n[Eval] ASBS-M: %d batches of %d samples\n', ev.n_batches, N_test);
pool_A = q_final;
t_start = tic;
for b = 2:ev.n_batches
    traj_b  = simulate_trajectories(netU, sample_source(N_test, params), params);
    pool_A  = [pool_A, traj_b(:,:, end)]; %#ok<AGROW>
end
time_A = toc(t_start) / max(ev.n_batches - 1, 1);

% Check analytic gradient against autodiff
q_chk = pool_A(:, 1:min(20, size(pool_A, 2)));
[g_ad, E_ad] = compute_energy_grad(q_chk, beta, params);
[E_an, g_an] = energy_grad_analytic(q_chk, beta, params);
fprintf('[Eval] analytic vs autodiff: max rel. diff energy %.1e, gradient %.1e\n', ...
    max(abs(E_an - E_ad)) / max(abs(E_ad)), max(abs(g_an - g_ad), [], 'all') / max(abs(g_ad), [], 'all'));

% HardFlow pools (one per lambda_oc)
n_lam  = numel(hf.lambda_oc);
pool_H = cell(1, n_lam);
time_H = zeros(1, n_lam);
for l = 1:n_lam
    fprintf('[Eval] HardFlow, lambda_oc = %g\n', hf.lambda_oc(l));
    t_start = tic;
    pool_H{l} = hardflow_sample(ev.n_batches * N_test, hf.lambda_oc(l), beta, params, hf);
    time_H(l) = toc(t_start) / ev.n_batches;
end

% Reference MCMC
ref = [];
if mc.run
    fprintf('[Eval] Reference constrained MCMC (%d chains x %d steps)\n', mc.n_chains, mc.n_steps);
    init = [pool_A, pool_H{:}];
    init = init(:, randperm(size(init, 2), min(mc.n_chains, size(init, 2))));
    t_start = tic;
    [ref_Q, ref_info] = constrained_mcmc(init, beta, params, mc);
    ref.E   = energy_grad_analytic(ref_Q, beta, params);
    ref.cls = homotopy_class(ref_Q, params);
    fprintf('[Eval] reference: acceptance %.2f, step %.3f, chains that changed class %.1f%%, %.0f s\n', ...
        ref_info.acc, ref_info.step, 100*ref_info.class_switch, toc(t_start));
    cls_lbl = {'U/low', 'U/mid', 'U/high', 'L/low', 'L/mid', 'L/high'};
    for c = 1:6
        if any(ref.cls == c)
            Ec = ref.E(ref.cls == c);
            fprintf('         class %-6s: reference beta*E min %.2f, median %.2f, share > min+30: %.2f (%d samples)\n', ...
                cls_lbl{c}, min(Ec), median(Ec), mean(Ec > min(Ec) + 30), numel(Ec));
        end
    end
end

% Metrics
names = [{'ASBS-M'}, arrayfun(@(x) sprintf('HardFlow(%g)', x), hf.lambda_oc, 'UniformOutput', false)];
pools = [{pool_A}, pool_H];
times = [time_A, time_H];
MET = cell(1, numel(pools));
for m = 1:numel(pools)
    MET{m} = robot_metrics(pools{m}, beta, params, ev, N_test, ref);
    MET{m}.time = times(m);
end

rows = {
    'Max constraint violation |c|_inf', @(M) sprintf('%.1e', M.viol_max)
    'Valid configurations (%)',          @(M) sprintf('%.1f', 100*M.feasible)
    'Collision-free (%)',                @(M) sprintf('%.1f', 100*M.collision_free)
    'Median clearance',                  @(M) sprintf('%.3f', M.clearance_med)
    'Energy beta*E: mean +- sd',         @(M) sprintf('%.2f +- %.2f', M.E_mean, M.E_sd)
    'Energy beta*E: median',             @(M) sprintf('%.2f', M.E_median)
    'Best energy in pool',               @(M) sprintf('%.3f', M.E_best_pool)
    sprintf('E[best of %d]', N_test),    @(M) sprintf('%.3f', M.E_best_batch)
    'P(above centre obstacle) [0.5]',    @(M) sprintf('%.3f', M.p_up)
    'Mode-balance z-score [|z|<2]',      @(M) sprintf('%.2f', M.z_up)
    'Mirror-symmetry p-value [>0.05]',   @(M) sprintf('%.3g', M.sym_p)
    'Mass in U/high + L/low (%) [100]',  @(M) sprintf('%.1f', 100*M.main)
    '  ... in their lowest basin (%) [100]', @(M) sprintf('%.1f', 100*M.target)
    'U/high share of these [0.5]',       @(M) sprintf('%.3f', M.bal)
    'Energy KS vs exact law [0]',        @(M) sprintf('%.3f', M.KS_all)
    'Homotopy classes visited',          @(M) sprintf('%d', M.n_classes)
    'Class entropy (bits)',              @(M) sprintf('%.2f', M.class_entropy)
    'Diversity (RMS joint distance)',    @(M) sprintf('%.3f', M.diversity)
    'Within-class energy KS vs ref.',    @(M) sprintf('%.3f (%.0f%%)', M.cond_KS, 100*M.cond_KS_mass)
    sprintf('Time per batch of %d [s]', N_test), @(M) sprintf('%.2f', M.time)
    };
fprintf('\n=================== SAMPLE QUALITY (pool of %d samples per method) ===================\n', ev.n_batches*N_test);
fprintf('%-36s', 'Metric');  fprintf(' | %-16s', names{:});  fprintf('\n');
fprintf('%s\n', repmat('-', 1, 36 + 19*numel(names)));
for r = 1:size(rows, 1)
    fprintf('%-36s', rows{r, 1});
    for m = 1:numel(MET), fprintf(' | %-16s', rows{r, 2}(MET{m})); end
    fprintf('\n');
end
fprintf(['[.] = value for an exact sampler of pi. KS column: weighted within-class KS distance to the\n' ...
    'reference MCMC (lower is better) and the share of samples in classes where it could be computed.\n' ...
    '[v2] pi puts (numerically) all its mass in U/high and L/low, 1/2 each: an exact sampler visits 2\n' ...
    'classes, class entropy 1 bit; more classes or entropy means mass in high-energy classes.\n']);
fprintf('ASBS-M training time: not timed here (section 3). HardFlow needs no training.\n');

%% 6. Figures
c_asbs = [0, 51, 102]/255;   c_hf = [217, 89, 0]/255;   c_ref = [0.45 0.45 0.45];
c_blue = [0, 0.4470, 0.7410];  c_red = [0.8500, 0.3250, 0.0980];
[~, l_star] = min(cellfun(@(M) M.E_best_batch, MET(2:end)));
MH = MET{1 + l_star};   MA = MET{1};
lbl_hf = sprintf('HardFlow ($\\lambda_{oc}=%g$)', hf.lambda_oc(l_star));

% Best configuration in a batch; expected best energy vs batch size
b1 = 1:N_test;
EA1 = MA.E(b1);  EA1(~MA.valid(b1)) = Inf;  [eA, iA] = min(EA1);
EH1 = MH.E(b1);  EH1(~MH.valid(b1)) = Inf;  [eH, iH] = min(EH1);
fig2 = figure('Units', 'inches', 'Position', [1, 1, 13, 5], 'Color', 'w');
tl2 = tiledlayout(fig2, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl2); hold(ax, 'on');
plot_workspace(ax, params);
hA = plot_robot(ax, pool_A(:, iA), params, c_asbs, 3);
hH = plot_robot(ax, pool_H{l_star}(:, iH), params, c_hf, 3);
legend(ax, [hA, hH], {sprintf('ASBS-M: $\\beta E = %.2f$', eA), sprintf('%s: $\\beta E = %.2f$', lbl_hf, eH)}, ...
    'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northwest', 'Box', 'off');
title(ax, sprintf('Best configuration in a batch of %d samples', N_test), 'Interpreter', 'latex', 'FontSize', 15);
xlabel(ax, '$X$ Workspace', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax, '$Y$ Workspace', 'Interpreter', 'latex', 'FontSize', 13);

ax = nexttile(tl2); hold(ax, 'on');
n_list = unique(round(logspace(0, log10(numel(MA.E)), 40)));
h = [];
for l = 1:n_lam
    if l == l_star, continue; end
    hl = plot(ax, n_list, expected_best_of_n(MET{1+l}.E(MET{1+l}.valid), n_list), '-', ...
        'Color', c_hf + 0.6*(1 - c_hf), 'LineWidth', 1);
end
h(end+1) = plot(ax, n_list, expected_best_of_n(MA.E(MA.valid), n_list), '-o', 'Color', c_asbs, ...
    'LineWidth', 2.2, 'MarkerSize', 4, 'MarkerFaceColor', c_asbs);
h(end+1) = plot(ax, n_list, expected_best_of_n(MH.E(MH.valid), n_list), '-s', 'Color', c_hf, ...
    'LineWidth', 2.2, 'MarkerSize', 4, 'MarkerFaceColor', c_hf);
if n_lam > 1, h(end+1) = hl; end
yl = ylim(ax);
plot(ax, N_test*[1 1], yl, ':', 'Color', 'k', 'LineWidth', 1.2);
ylim(ax, yl);
set(ax, 'XScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on');
grid(ax, 'on');
leg_txt = {'ASBS-M', lbl_hf};
if n_lam > 1, leg_txt{end+1} = 'HardFlow (other $\lambda_{oc}$)'; end
legend(ax, h, leg_txt, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast', 'Box', 'off');
xlabel(ax, 'Batch size $n$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax, '$\mathbf{E}[\min_{i \le n} \beta E(q_i)]$', 'Interpreter', 'latex', 'FontSize', 13);
title(ax, 'Expected best energy vs batch size', 'Interpreter', 'latex', 'FontSize', 15);

% One batch per method, coloured by mode
fig3 = figure('Units', 'inches', 'Position', [1, 1, 13, 5], 'Color', 'w');
tl3 = tiledlayout(fig3, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sets = {pool_A(:, b1), pool_H{l_star}(:, b1)};   ttl = {'ASBS-M', lbl_hf};   Ms = {MA, MH};
for s = 1:2
    ax = nexttile(tl3); hold(ax, 'on');
    plot_workspace(ax, params);
    up = Ms{s}.cls(b1) <= 3;
    for i = 1:N_test
        if up(i), col = c_blue; else, col = c_red; end
        plot_robot(ax, sets{s}(:, i), params, col + 0.7*(1 - col), 1);
    end
    title(ax, sprintf('%s: %d samples, %.0f\\%% above centre obstacle (exact: 50\\%%)', ttl{s}, N_test, 100*mean(up)), ...
        'Interpreter', 'latex', 'FontSize', 13);
end

% Energy law, homotopy classes, clearance
fig4 = figure('Units', 'inches', 'Position', [1, 1, 15, 4.5], 'Color', 'w');
tl4 = tiledlayout(fig4, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl4); hold(ax, 'on');
cls_names = {'U/low', 'U/mid', 'U/high', 'L/low', 'L/mid', 'L/high'};
h = [];  leg_txt = {'ASBS-M', lbl_hf};
if isempty(ref)
    h(end+1) = plot_ecdf(ax, MA.E, c_asbs, '-');
    h(end+1) = plot_ecdf(ax, MH.E, c_hf, '-');
    ttl_E = 'Energy distribution';
else
    cnt = accumarray(MA.cls(:), 1, [6, 1]);
    cnt(accumarray(ref.cls(:), 1, [6, 1]) < ev.ks_min(2)) = 0;
    [~, c_star] = max(cnt);
    h(end+1) = plot_ecdf(ax, MA.E(MA.cls == c_star), c_asbs, '-');
    if any(MH.cls == c_star), h(end+1) = plot_ecdf(ax, MH.E(MH.cls == c_star), c_hf, '-');
    else, leg_txt(2) = []; end
    h(end+1) = plot_ecdf(ax, ref.E(ref.cls == c_star), c_ref, '--');
    leg_txt{end+1} = 'Exact reference (MCMC)';
    ttl_E = sprintf('Energy within class %s', cls_names{c_star});
end
set(ax, 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on');  grid(ax, 'on');
legend(ax, h, leg_txt, 'Interpreter', 'latex', 'Location', 'southeast', 'Box', 'off');
xlabel(ax, '$\beta E(q)$', 'Interpreter', 'latex', 'FontSize', 13);  ylabel(ax, 'ECDF', 'FontSize', 13);
title(ax, ttl_E, 'Interpreter', 'none', 'FontSize', 14);

ax = nexttile(tl4); hold(ax, 'on');
bar(ax, (1:6) - 0.18, MA.class_frac, 0.34, 'FaceColor', c_asbs, 'EdgeColor', 'none');
bar(ax, (1:6) + 0.18, MH.class_frac, 0.34, 'FaceColor', c_hf, 'EdgeColor', 'none');
set(ax, 'XTick', 1:6, 'XTickLabel', cls_names, 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on');
grid(ax, 'on');
legend(ax, {'ASBS-M', lbl_hf}, 'Interpreter', 'latex', 'Location', 'northeast', 'Box', 'off');
ylabel(ax, 'Fraction of samples', 'FontSize', 13);
title(ax, {'Homotopy classes (U/L: above/below centre obstacle;', 'low/mid/high: passage at x = 5). Mirror pairs must match.'}, ...
    'FontSize', 11, 'FontWeight', 'normal');

ax = nexttile(tl4); hold(ax, 'on');
h = [plot_ecdf(ax, MA.clr, c_asbs, '-'), plot_ecdf(ax, MH.clr, c_hf, '-')];
yl = ylim(ax);  plot(ax, [0 0], yl, ':', 'Color', 'k', 'LineWidth', 1.2);  ylim(ax, yl);
set(ax, 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on');  grid(ax, 'on');
legend(ax, h, {'ASBS-M', lbl_hf}, 'Interpreter', 'latex', 'Location', 'southeast', 'Box', 'off');
xlabel(ax, 'Minimum clearance to obstacles (links vs. disks)', 'FontSize', 13);  ylabel(ax, 'ECDF', 'FontSize', 13);
title(ax, 'Safety margin (left of 0 = collision)', 'Interpreter', 'latex', 'FontSize', 14);

%% Helper functions

function [c, J, x_j, y_j] = robot_kinematics_and_jacobian(q, params)
    % Forward kinematics, constraints c(q) and Jacobian
    B = size(q, 2);
    d = params.d; L = params.L;
    theta = cumsum(q, 1);

    x_j = zeros(d, B); y_j = zeros(d, B);
    for i = 1:d
        if i == 1
            x_j(i, :) = L(i) * cos(theta(i,:));
            y_j(i, :) = L(i) * sin(theta(i,:));
        else
            x_j(i, :) = x_j(i-1, :) + L(i) * cos(theta(i,:));
            y_j(i, :) = y_j(i-1, :) + L(i) * sin(theta(i,:));
        end
    end

    % Constraints (m = 3), angle wrapped with atan2
    x_tip = x_j(d, :); y_tip = y_j(d, :); theta_tip = theta(d, :);
    c1 = x_tip - params.xT;
    c2 = y_tip - params.yT;
    c3 = atan2(sin(theta_tip - params.thetaT), cos(theta_tip - params.thetaT));
    c = [c1; c2; c3];

    % Jacobian (3 x d)
    J = zeros(3, d, B);
    for k = 1:d
        if k == 1
            x_prev = zeros(1, B); y_prev = zeros(1, B);
        else
            x_prev = x_j(k-1, :); y_prev = y_j(k-1, :);
        end
        J(1, k, :) = -(y_tip - y_prev);
        J(2, k, :) =  (x_tip - x_prev);
        J(3, k, :) =  1;
    end
end

function q_new = Retraction(q, params)
    % Newton projection onto c(q) = 0
    max_iters = 50; tol = 1e-4;
    q_curr = q;
    B = size(q_curr, 2);

    for iter = 1:max_iters
        [c, J] = robot_kinematics_and_jacobian(q_curr, params);
        if max(abs(c), [], 'all') < tol
            break;
        end
        c_mat = reshape(c, 3, 1, B);
        J_T = permute(J, [2, 1, 3]);
        JJT = pagemtimes(J, J_T) + 1e-5 * repmat(eye(3), 1, 1, B); % damping
        step = pagemtimes(J_T, pagemtimes(pageinv(JJT), c_mat));
        q_curr = q_curr - reshape(step, params.d, B);
    end
    q_new = q_curr;
end

function Pv = batchProjector(q, v, params)
    % Projection onto T_q M
    [~, J_num] = robot_kinematics_and_jacobian(q, params);
    B = size(q, 2);

    J_T = permute(J_num, [2, 1, 3]);
    JJT = pagemtimes(J_num, J_T) + 1e-6 * repmat(eye(3), 1, 1, B);
    Proj_comp = pagemtimes(J_T, pagemtimes(pageinv(JJT), J_num));
    P_mat = repmat(eye(params.d), 1, 1, B) - Proj_comp;

    if isdlarray(v)
        Pv_dl = pagemtimes(dlarray(P_mat), reshape(v, params.d, 1, B));
        Pv = reshape(Pv_dl, params.d, B);
    else
        Pv_num = pagemtimes(P_mat, reshape(v, params.d, 1, B));
        Pv = reshape(Pv_num, params.d, B);
    end
end

function [gradE, E_val] = compute_energy_grad(q_batch, beta, params)
    % beta*E and its gradient (autodiff)
    q_dl = dlarray(q_batch, 'CB');
    [E_val, gradE] = dlfeval(@energy_and_grad, q_dl, beta, params);
    gradE = extractdata(gradE);
    E_val = extractdata(E_val);
end

function [E_val, gradE] = energy_and_grad(q_dl, beta, params)
    B = size(q_dl, 2);

    % Loop instead of cumsum (dlarray support)
    theta = q_dl;
    for i = 2:params.d
        theta(i,:) = theta(i-1,:) + q_dl(i,:);
    end

    L = params.L;

    x_j = zeros(params.d, B, 'like', q_dl);
    y_j = zeros(params.d, B, 'like', q_dl);
    for i = 1:params.d
        if i == 1
            x_j(i, :) = L(i) * cos(theta(i,:));
            y_j(i, :) = L(i) * sin(theta(i,:));
        else
            x_j(i, :) = x_j(i-1, :) + L(i) * cos(theta(i,:));
            y_j(i, :) = y_j(i-1, :) + L(i) * sin(theta(i,:));
        end
    end

    E = zeros(1, B, 'like', q_dl);
    % Obstacle repulsion
    for k = 1:size(params.obs, 1)
        dist_sq = (x_j - params.obs(k,1)).^2 + (y_j - params.obs(k,2)).^2;
        E = E + sum(params.alpha * exp(-dist_sq / (2 * params.sigma_obs^2)), 1);
    end
    % Weak prior
    E = E + params.lambda * sum((q_dl - params.q_rest).^2, 1);

    E_beta = beta * E;
    gradE = dlgradient(sum(E_beta), q_dl);
    E_val = E_beta;
end

function [gradients, loss] = computeLossU(netU, X_batch, t_batch, V_target, sigma, BatchSize, params)
    U_pred = net_output(netU, X_batch, t_batch, params, true);
    U_proj = batchProjector(X_batch, U_pred, params);
    V_dl = dlarray(V_target, 'CB');

    residuals = U_proj + sigma .* V_dl;
    if ~isfield(params, 'loss_weight') || params.loss_weight
        loss = sum((residuals ./ sigma).^2, 'all') / BatchSize;  % 1/sigma_t^2 weight
    else
        loss = sum(residuals.^2, 'all') / BatchSize;             % unweighted
    end
    gradients = dlgradient(loss, netU.Learnables);
end

function [gradients, loss] = computeLossH(netH, X_N, b_target, d, BatchSize, params)
    H_pred = net_output(netH, X_N, [], params, true);
    H_proj = batchProjector(X_N, H_pred, params);
    b_dl = dlarray(b_target, 'CB');

    residuals = H_proj - b_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
    gradients = dlgradient(loss, netH.Learnables);
end

function q_traj = simulate_trajectories(net, q0, params)
    B = size(q0, 2);
    q_traj = zeros(params.d, B, params.N_steps + 1);
    q_traj(:,:,1) = q0;

    for j = 1:params.N_steps
        t = (j-1) * params.dt;
        q_curr = q_traj(:,:,j);

        u_val = predict_net(net, q_curr, params.d, B, true, t, params);
        noise = randn(params.d, B);

        s = params.sigma(min(j, numel(params.sigma)));
        drift = s * batchProjector(q_curr, u_val, params) * params.dt;
        diff_term = s * sqrt(params.dt) * batchProjector(q_curr, noise, params);

        step_update = q_curr + drift + diff_term;
        q_traj(:,:,j+1) = Retraction(step_update, params); % retraction onto M
    end
end

function out = predict_net(net, X, d, B, has_time, t, params)
    if has_time, tt = repmat(t, 1, B); else, tt = []; end
    out = extractdata(net_output(net, X, tt, params, false));
end

function out = net_output(net, X, t, params, train)
    % Network output; odd part [NN(q) - NN(-q)]/2 if params.mirror
    B = size(X, 2);
    if isfield(params, 'mirror') && params.mirror
        inputs = dlarray([[X; t], [-X; t]], 'CB');
    else
        inputs = dlarray([X; t], 'CB');
    end
    if train, o = forward(net, inputs); else, o = predict(net, inputs); end
    if isfield(params, 'mirror') && params.mirror
        out = 0.5*(o(:, 1:B) - o(:, B+1:end));
    else
        out = o;
    end
end

function q0 = sample_source(B, params)
    % Source: Retraction(src_mu + src_sd*randn), mirrored with probability 1/2
    q0 = Retraction(params.src_mu + params.src_sd * randn(params.d, B), params);
    if params.mirror
        q0 = q0 .* (2*(rand(1, B) < 0.5) - 1);
    end
end

%% Baselines, reference MCMC, metrics, plotting

function [E, G] = energy_grad_analytic(q, beta, params)
    % beta*E with closed-form gradient
    L = params.L(:);
    theta = cumsum(q, 1);
    x_j = cumsum(L .* cos(theta), 1);
    y_j = cumsum(L .* sin(theta), 1);
    E  = params.lambda * sum((q - params.q_rest).^2, 1);
    gx = zeros(size(x_j));  gy = zeros(size(y_j));
    for k = 1:size(params.obs, 1)
        ek = params.alpha * exp(-((x_j - params.obs(k,1)).^2 + (y_j - params.obs(k,2)).^2) / (2*params.sigma_obs^2));
        E  = E + sum(ek, 1);
        gx = gx - ek .* (x_j - params.obs(k,1)) / params.sigma_obs^2;
        gy = gy - ek .* (y_j - params.obs(k,2)) / params.sigma_obs^2;
    end
    E = beta * E;
    if nargout > 1
        Sx  = flipud(cumsum(flipud(gx), 1));
        Sy  = flipud(cumsum(flipud(gy), 1));
        dth = L .* (-sin(theta) .* Sx + cos(theta) .* Sy);
        G   = beta * (flipud(cumsum(flipud(dth), 1)) + 2 * params.lambda * (q - params.q_rest));
    end
end

function v = gauss_fm_velocity(x, t, hf)
    % Exact flow-matching velocity of the source (Gaussian or symmetric mixture), linear path
    sig2 = t^2 * hf.s^2 + (1 - t)^2;
    k    = (t * hf.s^2 - (1 - t)) / sig2;
    if ~isfield(hf, 'mirror') || ~hf.mirror
        v = hf.mu + k * (x - t * hf.mu);
        return
    end
    lw1 = -sum((x - t*hf.mu).^2, 1) / (2*sig2);    % posterior weights
    lw2 = -sum((x + t*hf.mu).^2, 1) / (2*sig2);
    w1  = 1 ./ (1 + exp(lw2 - lw1));
    v   = w1 .* (hf.mu + k*(x - t*hf.mu)) + (1 - w1) .* (-hf.mu + k*(x + t*hf.mu));
end

function q = hardflow_sample(N, lambda_oc, beta, params, hf)
    % HardFlow, Algorithm 1 with alpha_t = t, beta_t = 1 - t; Eq. (11) solved in the last steps.
    % Every output satisfies c(q) = 0.
    x  = randn(params.d, N);
    tg = linspace(0, 1, hf.N_flow + 1);
    for i = 1:hf.N_flow
        t = tg(i);  t1 = tg(i+1);  dt = t1 - t;
        xb  = x + gauss_fm_velocity(x, t, hf) * dt;
        v1  = gauss_fm_velocity(xb, t1, hf);
        xNb = xb + (1 - t1) * v1;
        if i > round(hf.N_flow * (1 - hf.active))
            xNs = solve_eq11(xNb, lambda_oc * t1^2 / dt, beta, params, hf);
        else
            xNs = xNb;
        end
        x = t1 * xNs + (1 - t1) * (xb - t1 * v1);
    end
    q = x;
end

function y = solve_eq11(y0, m_reg, beta, params, hf)
    % min_y beta*E(y) + m_reg/2 |y - y0|^2 s.t. c(y) = 0 (projected gradient + Armijo)
    B = size(y0, 2);
    y = Retraction(y0, params);
    [Cv, g] = energy_grad_analytic(y, beta, params);
    f   = Cv + 0.5 * m_reg * sum((y - y0).^2, 1);
    eta = hf.eta0 * ones(1, B);
    for it = 1:hf.max_iter
        PG  = batchProjector(y, g + m_reg * (y - y0), params);
        gn2 = sum(PG.^2, 1);
        act = gn2 > hf.tol^2;
        if ~any(act), break; end
        done = false(1, B);
        for bt = 1:hf.max_bt
            ii = find(act & ~done);
            if isempty(ii), break; end
            yt = Retraction(y(:, ii) - eta(ii) .* PG(:, ii), params);
            ct = robot_kinematics_and_jacobian(yt, params);
            ft = energy_grad_analytic(yt, beta, params) + 0.5 * m_reg * sum((yt - y0(:, ii)).^2, 1);
            ft(max(abs(ct), [], 1) > 1e-4) = Inf;          % reject failed retractions
            ok = ft <= f(ii) - 1e-4 * eta(ii) .* gn2(ii);
            y(:, ii(ok)) = yt(:, ok);
            f(ii(ok))    = ft(ok);
            done(ii(ok)) = true;
            eta(ii(~ok)) = eta(ii(~ok)) / 2;
        end
        if any(done)
            [~, g(:, done)] = energy_grad_analytic(y(:, done), beta, params);
            eta(done) = min(2 * eta(done), hf.eta_max);
        end
    end
end

function [snaps, info] = constrained_mcmc(Q, beta, params, mc)
    % Constrained MCMC on M (Zappa, Holmes-Cerfon & Goodman, 2018, Alg. 1)
    [d, C] = size(Q);
    Q   = tight_projection(Q, params, mc.newton_tol, 100);
    cls0 = homotopy_class(Q, params);
    E   = energy_grad_analytic(Q, beta, params);
    log_s = log(mc.step0);
    n_snap = floor((mc.n_steps - mc.burn) / mc.thin);
    snaps  = zeros(d, C * n_snap);  ns = 0;
    acc_hist = zeros(mc.n_steps, 1);
    for k = 1:mc.n_steps
        s = exp(log_s);
        [~, Jq] = robot_kinematics_and_jacobian(Q, params);
        V = s * tangent_projection(Jq, randn(d, C));
        [Y, ok_f] = normal_newton(Q + V, Jq, params, mc);
        Y(:, ~ok_f) = Q(:, ~ok_f);
        [~, Jy] = robot_kinematics_and_jacobian(Y, params);
        Vr = tangent_projection(Jy, Q - Y);
        [Qr, ok_r] = normal_newton(Y + Vr, Jy, params, mc);
        ok_r = ok_r & (max(abs(Qr - Q), [], 1) < mc.rev_tol);
        Ey   = energy_grad_analytic(Y, beta, params);
        logr = -(Ey - E) - (sum(Vr.^2, 1) - sum(V.^2, 1)) / (2 * s^2);
        acc  = ok_f & ok_r & isfinite(logr) & (log(rand(1, C)) < logr);
        Q(:, acc) = Y(:, acc);
        E(acc)    = Ey(acc);
        acc_hist(k) = mean(acc);
        if k <= mc.burn
            log_s = log_s + (acc_hist(k) - mc.target_acc) / k^0.6;
        elseif mod(k - mc.burn, mc.thin) == 0
            snaps(:, ns*C + (1:C)) = Q;
            ns = ns + 1;
        end
    end
    snaps = snaps(:, 1:ns*C);
    info.acc  = mean(acc_hist(mc.burn+1:end));
    info.step = exp(log_s);
    info.class_switch = mean(homotopy_class(Q, params) ~= cls0);
end

function PV = tangent_projection(J, V)
    % Exact projection onto ker(J)
    [d, C] = size(V);
    JT = permute(J, [2 1 3]);
    W  = pagemtimes(pageinv(pagemtimes(J, JT)), pagemtimes(J, reshape(V, d, 1, C)));
    PV = V - reshape(pagemtimes(JT, W), d, C);
end

function [Y, ok] = normal_newton(Z, Jx, params, mc)
    % Newton along the normal space at x
    [d, C] = size(Z);
    JxT = permute(Jx, [2 1 3]);
    a = zeros(3, 1, C);
    for it = 1:mc.newton_iters
        Y = Z + reshape(pagemtimes(JxT, a), d, C);
        [c, Jy] = robot_kinematics_and_jacobian(Y, params);
        if all(max(abs(c), [], 1) < mc.newton_tol), break; end
        a = a - pagemtimes(pageinv(pagemtimes(Jy, JxT)), reshape(c, 3, 1, C));
    end
    Y  = Z + reshape(pagemtimes(JxT, a), d, C);
    c  = robot_kinematics_and_jacobian(Y, params);
    ok = (max(abs(c), [], 1) < mc.newton_tol) & all(isfinite(Y), 1);
end

function q = tight_projection(q, params, tol, iters)
    % Undamped Newton projection onto c(q) = 0
    [d, B] = size(q);
    for it = 1:iters
        [c, J] = robot_kinematics_and_jacobian(q, params);
        if max(abs(c), [], 'all') < tol, break; end
        JT = permute(J, [2 1 3]);
        q  = q - reshape(pagemtimes(JT, pagemtimes(pageinv(pagemtimes(J, JT)), reshape(c, 3, 1, B))), d, B);
    end
end

function cls = homotopy_class(q, params)
    % Classes 1..6 = [U/low U/mid U/high L/low L/mid L/high]; mirror pairs 1-6, 2-5, 3-4
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q, params);
    B = size(q, 2);
    X = [zeros(1, B); x_j];  Y = [zeros(1, B); y_j];
    yc = nan(2, B);
    xs = [3, 5];
    for s = 1:2
        for k = 1:params.d
            a = X(k, :) - xs(s);  b = X(k+1, :) - xs(s);
            hit = (a .* b <= 0) & (a ~= b) & isnan(yc(s, :));
            w = a(hit) ./ (a(hit) - b(hit));
            yc(s, hit) = Y(k, hit) + w .* (Y(k+1, hit) - Y(k, hit));
        end
    end
    c5  = (yc(2, :) > 1) - (yc(2, :) < -1);          % -1 low, 0 mid, +1 high
    cls = 3 * (yc(1, :) <= 0) + c5 + 2;
end

function clr = clearance_robot(q, params, r_obs)
    % Minimum distance between links and obstacle disks
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q, params);
    B = size(q, 2);
    X = [zeros(1, B); x_j];  Y = [zeros(1, B); y_j];
    clr = inf(1, B);
    for o = 1:size(params.obs, 1)
        ox = params.obs(o, 1);  oy = params.obs(o, 2);
        for k = 1:params.d
            dx = X(k+1, :) - X(k, :);  dy = Y(k+1, :) - Y(k, :);
            t  = min(max(((ox - X(k, :)) .* dx + (oy - Y(k, :)) .* dy) ./ (dx.^2 + dy.^2), 0), 1);
            clr = min(clr, hypot(X(k, :) + t .* dx - ox, Y(k, :) + t .* dy - oy) - r_obs);
        end
    end
end

function M = robot_metrics(Q, beta, params, ev, N_batch, ref)
    n    = size(Q, 2);
    c    = robot_kinematics_and_jacobian(Q, params);
    viol = max(abs(c), [], 1);
    E    = energy_grad_analytic(Q, beta, params);
    cls  = homotopy_class(Q, params);
    clr  = clearance_robot(Q, params, ev.r_obs);
    valid = viol <= ev.feas_tol;
    M.n = n;
    M.viol_max = max(viol);
    M.feasible = mean(valid);
    M.collision_free = mean(clr > 0);
    M.clearance_med  = median(clr);
    M.E_mean = mean(E);  M.E_sd = std(E);  M.E_median = median(E);
    M.E_best_pool  = min(E(valid));
    M.E_best_batch = expected_best_of_n(E(valid), N_batch);
    M.p_up = mean(cls <= 3);
    M.z_up = (M.p_up - 0.5) / sqrt(0.25 / n);
    counts = accumarray(cls(:), 1, [6, 1])';
    M.main = mean(cls == 3 | cls == 4);                    % two lowest classes
    M.bal  = counts(3) / max(counts(3) + counts(4), 1);    % share of U/high (exact 0.5)
    M.target = NaN;
    M.class_frac = counts / n;
    [M.sym_chi2, M.sym_p] = mirror_symmetry_test(counts);
    M.n_classes = nnz(counts);
    pf = counts(counts > 0) / n;
    M.class_entropy = -sum(pf .* log2(pf));
    M.diversity = mean_pairwise_rms(Q(:, 1:min(n, ev.div_max)), params);
    if isempty(ref)
        M.cond_KS = NaN;  M.cond_KS_mass = NaN;  M.KS_all = NaN;
    else
        [M.cond_KS, M.cond_KS_mass] = conditional_ks(E, cls, ref.E, ref.cls, ev.ks_min);
        % Exact energy law: reference restricted to the lowest basin
        main_ref = ref.cls == 3 | ref.cls == 4;
        if any(main_ref)
            E_lo = min(ref.E(main_ref)) + 30;
            low  = main_ref & ref.E <= E_lo;
            M.KS_all = ks2(E, ref.E(low));
            M.target = mean((cls == 3 | cls == 4) & E <= E_lo);
        else
            M.KS_all = NaN;
        end
    end
    M.E = E;  M.cls = cls;  M.clr = clr;  M.valid = valid;
end

function eb = expected_best_of_n(E, n_list)
    % E[min of n draws without replacement from E]
    Es = sort(E(:));  M = numel(Es);  k = (1:M)';
    eb = nan(size(n_list));
    for i = 1:numel(n_list)
        n = n_list(i);
        if n > M || n < 1, continue; end
        v  = (M - k) >= (n - 1);
        kv = k(v);
        lp = gammaln(M - kv + 1) - gammaln(n) - gammaln(M - kv - n + 2) ...
           - (gammaln(M + 1) - gammaln(n + 1) - gammaln(M - n + 1));
        eb(i) = sum(exp(lp) .* Es(v));
    end
end

function [chi2, p] = mirror_symmetry_test(counts)
    % Chi-square test: mirrored classes equally likely
    pairs = [1 6; 2 5; 3 4];
    na = counts(pairs(:, 1));  nb = counts(pairs(:, 2));
    use = (na + nb) > 0;
    chi2 = sum((na(use) - nb(use)).^2 ./ (na(use) + nb(use)));
    p = gammainc(chi2 / 2, nnz(use) / 2, 'upper');
end

function r = mean_pairwise_rms(Q, params)
    % Mean pairwise RMS joint distance
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(Q, params);
    P  = [x_j; y_j];
    n  = size(P, 2);
    sq = sum(P.^2, 1);
    D2 = max(sq' + sq - 2 * (P' * P), 0);
    r  = sum(sqrt(D2(:) / params.d)) / (n * (n - 1));
end

function [ks_w, mass] = conditional_ks(E, cls, Eref, cls_ref, min_n)
    % Within-class KS to the reference, weighted by class mass
    ks = nan(1, 6);  w = zeros(1, 6);
    for c = 1:6
        a = E(cls == c);  b = Eref(cls_ref == c);
        if numel(a) >= min_n(1) && numel(b) >= min_n(2)
            ks(c) = ks2(a, b);  w(c) = numel(a);
        end
    end
    mass = sum(w) / numel(E);
    if mass == 0, ks_w = NaN; else, ks_w = sum(w(w > 0) .* ks(w > 0)) / sum(w); end
end

function D = ks2(a, b)
    % Two-sample KS statistic
    a = a(:);  b = b(:);
    [vs, idx] = sort([a; b]);
    lab = [ones(numel(a), 1) / numel(a); -ones(numel(b), 1) / numel(b)];
    cs  = cumsum(lab(idx));
    D   = max(abs(cs([diff(vs) ~= 0; true])));
end

function plot_workspace(ax, params)
    theta_circ = linspace(0, 2*pi, 100);
    for k = 1:size(params.obs, 1)
        fill(ax, params.obs(k,1) + params.sigma_obs*cos(theta_circ), params.obs(k,2) + params.sigma_obs*sin(theta_circ), ...
            [0.7 0.7 0.7], 'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 1.2, 'FaceAlpha', 0.5);
    end
    plot(ax, params.xT, params.yT, 'ks', 'MarkerSize', 11, 'MarkerFaceColor', [1 0.84 0], 'LineWidth', 1.5);
    plot(ax, 0, 0, 'k^', 'MarkerSize', 12, 'MarkerFaceColor', [0.83 0.83 0.83], 'LineWidth', 1.5);
    axis(ax, 'equal');  box(ax, 'on');
    xlim(ax, [-1, 10]);  ylim(ax, [-3, 5]);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 12);
end

function h = plot_robot(ax, q, params, col, lw)
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q, params);
    h = plot(ax, [0; x_j], [0; y_j], '-o', 'Color', col, 'LineWidth', lw, 'MarkerSize', 2 + lw, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', col);
end

function h = plot_ecdf(ax, v, col, style)
    v = sort(v(:));
    h = stairs(ax, v, (1:numel(v))' / numel(v), style, 'Color', col, 'LineWidth', 1.8);
end
