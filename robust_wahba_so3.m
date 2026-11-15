% Extended R-ASBS (Alg. 2) for robust Wahba on SO(3) via S^3 quaternions, compared with
% R-ASBS (Alg. 1), Riemannian MALA, multistart RGD, uniform sampling and the exact target.
% ("ASBS-M" in the code = R-ASBS / Extended R-ASBS in the paper)

clear; close all; clc;

%% 1. Base problem and algorithm configuration
baseParams = default_wahba_params();

%% 2. Experiment grid
expCfg.betas = [0.25 0.50 1.00 2.00 4.00];
expCfg.NsList = [500 1000 2500 5000 10000];
expCfg.nRepeats = 5;
expCfg.outlierRatios = [0.25 0.50 0.75 0.85 0.90 0.95];
expCfg.seed0 = 1200;
expCfg.saveCsv = true;
expCfg.makePlots = false;
expCfg.sameInstanceAcrossBeta = true;   % same instance for every beta

%% 2b. Method comparison
cmpCfg.runRASBS  = true;                 % R-ASBS (exact S^3 geometry)
cmpCfg.runMALA   = true;                 % Riemannian MALA
cmpCfg.runRGD    = true;                 % multistart RGD
cmpCfg.runTarget = true;                 % exact target around q*
cmpCfg.rasbs_nt  = 8;                    % bridge times per pair
cmpCfg.mala_budget_mult = [1 4 16];      % MALA budget / ASBS-M training budget
cmpCfg.mala_delta0      = 1e-3;          % initial MALA step (adapted)
cmpCfg.mala_target_acc  = 0.574;
cmpCfg.rgd_K      = [1 10 100 1000];     % refined starts
cmpCfg.ref_chains = 2000;                % reference chains at q*
cmpCfg.ref_steps  = 400;
cmpCfg.deltas_deg = [1 2 5 10 20];       % radii around q* (deg)
cmpCfg.succ_deg   = 5;                   % success threshold (deg)
cmpCfg.n_curve    = [1 3 10 30 100 300 1000 3000 10000];   % batch sizes for the selection curve
assert(any(cmpCfg.deltas_deg == cmpCfg.succ_deg), 'cmpCfg.deltas_deg must contain cmpCfg.succ_deg');

%% 3. Run experiments
[T, T_cmp, cmpStore] = run_asbs_experiment_grid(baseParams, expCfg, cmpCfg);

disp(T);

if expCfg.saveCsv
    writetable(T, 'asbs_wahba_experiment_results.csv');
    fprintf('\nSaved results to asbs_wahba_experiment_results.csv\n');
    writetable(T_cmp, 'asbs_wahba_method_comparison.csv');
    save('asbs_wahba_method_comparison.mat', 'T', 'T_cmp', 'cmpStore', 'expCfg', 'cmpCfg');
    fprintf('Saved comparison to asbs_wahba_method_comparison.csv / .mat\n');
end

% Summary at the largest Ns
pc = strrep(sprintf('p_%gdeg', cmpCfg.succ_deg), '.', 'p');
Gs = groupsummary(T_cmp(T_cmp.Ns == max(expCfg.NsList) | strcmp(T_cmp.method, 'Target (exact, local)'), :), ...
    {'outlierRatio', 'beta', 'method', 'K'}, {'mean', 'median'}, {'success', pc, 'n95', 'KS_target', 'energyEvals'});
disp(Gs(:, {'outlierRatio', 'beta', 'method', 'K', 'mean_success', ['mean_' pc], 'median_n95', 'mean_KS_target', 'mean_energyEvals'}));

if expCfg.makePlots
    plot_asbs_experiment_results(T);
    plot_comparison_results(T_cmp, cmpStore, cmpCfg);
end

%% Experiment functions

function params = default_wahba_params()
    params.useSynthetic = true;

    % Synthetic instance
    params.Ncorr = 1000;
    params.outlierRatio = 0.75;
    params.inlierNoise = 0.10;
    params.sigmaMeas = 0.10;
    params.trueAxis = [0.35; -0.75; 0.56];
    params.trueAngleDeg = 72;

    % TLS objective
    params.cbar = 2.7955;
    params.beta = 1.0;
    params.useSmoothTLS = true;
    params.tlsSmoothing = 0.20;

    % ASBS-M
    params.d = 4;
    params.B = 512;
    params.N_steps = 150;
    params.dt = 1 / params.N_steps;
    params.sigma = 0.50;            % unused (sigma(t) schedule below)
    params.sig_max = 1.0;           % sigma(t) = sig_max^(1-t) sig_min^t
    params.sig_min = 0.02;
    params = noise_schedule(params);                                       % params.sig, params.V1
    params.K_epochs = 600;
    params.lr = 1e-3;
    params.lr_end = 0.1;            % cosine decay to lr*lr_end
    params.n_inner = 20;            % Adam steps per epoch
    params.n_sub = 10;              % minibatch = n_sub*B triples
    params.loss_weight = true;      % 1/sigma_t^2 loss weight (false = unweighted)
    params.gradClip = 50;
    params.hiddenWidth = 512;
    params.printEvery = 50;
    params.verbose = true;

    % Sampling
    params.N_test = 10000;
    params.N_random_baseline = 10000;
end

function [T, T_cmp, cmpStore] = run_asbs_experiment_grid(baseParams, expCfg, cmpCfg)
    T = table();
    T_cmp = table();
    cmpStore = struct([]);
    totalRuns = numel(expCfg.betas) * numel(expCfg.outlierRatios) * expCfg.nRepeats;
    runId = 0;

    for rep = 1:expCfg.nRepeats
        for outlierRatio = expCfg.outlierRatios
            for beta = expCfg.betas
                runId = runId + 1;
                params = baseParams;
                params.beta = beta;
                params.outlierRatio = outlierRatio;
                params.N_test = max(expCfg.NsList);
                params.N_random_baseline = max(expCfg.NsList);
                params.dt = 1 / params.N_steps;

                % One seed per setting
                runSeed = expCfg.seed0 + 100000 * rep + 1000 * round(100 * outlierRatio) + round(100 * beta);
                rng(runSeed);

                % Instance seed independent of beta
                if expCfg.sameInstanceAcrossBeta
                    params.instanceSeed = expCfg.seed0 + 100000 * rep + 1000 * round(100 * outlierRatio);
                end

                fprintf('\n============================================================\n');
                fprintf('Run %d / %d | repeat %d | beta %.3f | outliers %.2f\n', ...
                    runId, totalRuns, rep, beta, outlierRatio);
                fprintf('============================================================\n');

                [T_run, Tc_run, store_run] = run_single_asbs_setting(params, expCfg.NsList, rep, runSeed, cmpCfg);
                T = [T; T_run]; %#ok<AGROW>
                T_cmp = [T_cmp; Tc_run]; %#ok<AGROW>
                if isempty(cmpStore), cmpStore = store_run; else, cmpStore(end+1) = store_run; end %#ok<AGROW>
            end
        end
    end
end

function [T, T_cmp, store] = run_single_asbs_setting(params, NsList, rep, runSeed, cmpCfg)
    params = noise_schedule(params);
    if params.useSynthetic
        if isfield(params, 'instanceSeed')
            rng(params.instanceSeed);
            params = makeSyntheticWahba(params);
            rng(runSeed);
        else
            params = makeSyntheticWahba(params);
        end
    else
        % Real data
        params.A = normalizeColumns(params.A);  % 3 x N source unit vectors
        params.Bmeas = normalizeColumns(params.Bmeas);  % 3 x N measured unit vectors
        params.sigmas = params.sigmas(:)';  % noise scales
        params.Ncorr = size(params.A, 2);
        if ~isfield(params, 'inlierMask')
            params.inlierMask = true(1, params.Ncorr);
        end
    end

    fprintf('N = %d, cbar = %.2f, beta = %.3f, outlier ratio = %.2f\n', ...
        params.Ncorr, params.cbar, params.beta, mean(~params.inlierMask));

    trueTLS = NaN;
    if isfield(params, 'Rtrue')
        trueTLS = wahba_tls_energy(rotmToQuat(params.Rtrue), params, false);
        fprintf('True TLS objective: %.4f\n', trueTLS);
    end

    tTrain = tic;
    [netU, trainInfo] = train_asbs_m(params);
    trainTimeASBS = toc(tTrain);

    fprintf('\nSampling terminal rotations without local refinement...\n');
    maxNs = max(NsList);

    tSample = tic;
    q0_test = uniformQuaternions(maxNs);
    final_trajs = simulate_trajectories(netU, q0_test, params);
    q_samples = final_trajs(:, :, end);
    sampleTimeASBS = toc(tSample);
    F_samples = wahba_tls_energy(q_samples, params, false);

    q_random = uniformQuaternions(maxNs);
    F_random = wahba_tls_energy(q_random, params, false);

    T = summarize_sample_prefixes(params, NsList, q_samples, F_samples, q_random, F_random, ...
        trueTLS, rep, runSeed, trainInfo);

    % Comparison with the other methods
    costASBS.trainTime  = trainTimeASBS;
    costASBS.sampleTime = sampleTimeASBS;
    costASBS.trainEvals = params.K_epochs * params.B;      % grad E evaluations in training
    [T_cmp, store] = run_method_comparison(params, NsList, q_samples, q_random, rep, runSeed, cmpCfg, costASBS);
end

function [netU, trainInfo] = train_asbs_m(params)
    [netU, netH] = initialize_networks(params);

    avgU = []; avgSqU = [];
    avgH = []; avgSqH = [];
    [iterU, iterH] = deal(0);
    lossUHist = zeros(params.K_epochs, 1);
    lossHHist = zeros(params.K_epochs, 1);
    bestTrainHist = zeros(params.K_epochs, 1);
    M_all = params.B * params.N_steps;
    M_sub = min(M_all, params.n_sub * params.B);
    T_traj = kron((0:params.N_steps-1) * params.dt, ones(1, params.B));
    S_traj = kron(params.sig', ones(1, params.B));

    for k = 1:params.K_epochs
        lr_k = params.lr * (params.lr_end + (1 - params.lr_end) * 0.5 * (1 + cos(pi * (k - 1) / params.K_epochs)));
        q0 = uniformQuaternions(params.B);

        q_traj = simulate_trajectories(netU, q0, params);

        % Terminal adjoint v_N = P(grad E + h)
        q_N = q_traj(:, :, params.N_steps + 1);
        h_val = predict_net(netH, q_N, params.d, params.B, false, 0);
        [gradE, ~] = compute_energy_grad(q_N, params);
        v_curr = batchProjector(q_N, gradE + h_val, params);

        % Backward adjoint transport by projection
        V_traj = zeros(params.d, params.B, params.N_steps);
        for j = params.N_steps:-1:1
            q_j = q_traj(:, :, j);
            V_traj(:, :, j) = batchProjector(q_j, v_curr, params);
            v_curr = V_traj(:, :, j);
        end

        q_traj_flat = reshape(q_traj(:, :, 1:params.N_steps), params.d, M_all);
        V_traj_flat = reshape(V_traj, params.d, M_all);

        % Controller: n_inner Adam steps on random minibatches
        for it = 1:params.n_inner
            idx = randi(M_all, 1, M_sub);
            iterU = iterU + 1;
            [gradU, lossU] = dlfeval(@computeLossU, netU, q_traj_flat(:, idx), T_traj(idx), ...
                V_traj_flat(:, idx), S_traj(idx), M_sub, params);
            gradU = clipGradients(gradU, params.gradClip);
            [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, iterU, lr_k);
        end

        % Corrector: projected chord
        q_resampled = simulate_trajectories(netU, q0, params);
        q_1 = q_resampled(:, :, params.N_steps + 1);
        chord = (q_1 - q0) / params.V1;                          % total reference variance
        b_target = -batchProjector(q_1, chord, params);

        for it = 1:params.n_inner
            iterH = iterH + 1;
            [gradH, lossH] = dlfeval(@computeLossH, netH, q_1, b_target, params.d, params.B, params);
            gradH = clipGradients(gradH, params.gradClip);
            [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, iterH, lr_k);
        end

        lossUHist(k) = double(gather(extractdata(lossU)));
        lossHHist(k) = double(gather(extractdata(lossH)));
        bestTrainHist(k) = min(wahba_tls_energy(q_1, params, false));

        if params.verbose && (mod(k, params.printEvery) == 0 || k == 1)
            fprintf('Epoch %04d | Loss U %.4e | Loss H %.4e | best train TLS %.4f\n', ...
                k, lossUHist(k), lossHHist(k), bestTrainHist(k));
        end
    end

    trainInfo.lossUFinal = lossUHist(end);
    trainInfo.lossHFinal = lossHHist(end);
    trainInfo.bestTrainFinal = bestTrainHist(end);
    trainInfo.bestTrainOverall = min(bestTrainHist);
end

function params = noise_schedule(params)
    % sigma_j = sig_max^(1-t_j) sig_min^t_j, V1 = sum_j sigma_j^2 dt
    params.dt = 1 / params.N_steps;
    t_grid = (0:params.N_steps-1)' * params.dt;
    params.sig = params.sig_max.^(1 - t_grid) .* params.sig_min.^t_grid;
    params.V1 = sum(params.sig.^2) * params.dt;
end

function [netU, netH] = initialize_networks(params)
    layersU = [
        featureInputLayer(params.d + 1, 'Normalization', 'none')
        fullyConnectedLayer(params.hiddenWidth)
        tanhLayer
        fullyConnectedLayer(params.hiddenWidth)
        tanhLayer
        fullyConnectedLayer(params.d)
    ];
    netU = dlnetwork(layersU);

    layersH = [
        featureInputLayer(params.d, 'Normalization', 'none')
        fullyConnectedLayer(params.hiddenWidth)
        tanhLayer
        fullyConnectedLayer(params.hiddenWidth)
        tanhLayer
        fullyConnectedLayer(params.d)
    ];
    netH = dlnetwork(layersH);
end

function T = summarize_sample_prefixes(params, NsList, q_samples, F_samples, q_random, F_random, ...
    trueTLS, rep, runSeed, trainInfo)

    rows = cell(numel(NsList), 1);

    for jj = 1:numel(NsList)
        Ns = NsList(jj);

        [bestASBS, bestIdx] = min(F_samples(1:Ns));
        q_best = q_samples(:, bestIdx);

        [bestRandom, bestRandIdx] = min(F_random(1:Ns));
        q_best_random = q_random(:, bestRandIdx);

        bestASBSRotErrDeg = NaN;
        bestRandomRotErrDeg = NaN;
        medianSampleRotErrDeg = NaN;
        fracSamplesBelow5Deg = NaN;
        fracSamplesBelow10Deg = NaN;

        if isfield(params, 'Rtrue')
            R_asbs = quatToRotm(q_best);
            R_rand = quatToRotm(q_best_random);
            bestASBSRotErrDeg = rotationErrorDeg(R_asbs, params.Rtrue);
            bestRandomRotErrDeg = rotationErrorDeg(R_rand, params.Rtrue);

            sampleRotErrs = rotationErrorDegBatch(q_samples(:, 1:Ns), params.Rtrue);
            medianSampleRotErrDeg = median(sampleRotErrs);
            fracSamplesBelow5Deg = mean(sampleRotErrs <= 5);
            fracSamplesBelow10Deg = mean(sampleRotErrs <= 10);
        end

        residualInfo = wahba_residuals(q_best, params);
        active = residualInfo.s < params.cbar^2;
        activeCount = sum(active);
        trueInliersActive = NaN;
        trueOutliersClipped = NaN;
        if isfield(params, 'inlierMask')
            trueInliersActive = sum(active & params.inlierMask);
            trueOutliersClipped = sum((~active) & (~params.inlierMask));
        end

        rows{jj} = table( ...
            params.beta, params.outlierRatio, rep, runSeed, params.Ncorr, Ns, ...
            trueTLS, bestASBS, bestRandom, bestASBSRotErrDeg, bestRandomRotErrDeg, ...
            medianSampleRotErrDeg, fracSamplesBelow5Deg, fracSamplesBelow10Deg, ...
            activeCount, trueInliersActive, trueOutliersClipped, ...
            trainInfo.lossUFinal, trainInfo.lossHFinal, trainInfo.bestTrainFinal, trainInfo.bestTrainOverall, ...
            'VariableNames', { ...
                'beta', 'outlierRatio', 'repeat', 'seed', 'Ncorr', 'Ns', ...
                'trueTLS', 'bestASBS_TLS', 'bestRandom_TLS', 'bestASBS_rotErrDeg', 'bestRandom_rotErrDeg', ...
                'medianSample_rotErrDeg', 'fracSamplesBelow5Deg', 'fracSamplesBelow10Deg', ...
                'activeCount', 'trueInliersActive', 'trueOutliersClipped', ...
                'lossUFinal', 'lossHFinal', 'bestTrainFinal', 'bestTrainOverall'});

        fprintf('Ns %6d | ASBS TLS %.4f | random TLS %.4f | ASBS err %.3f deg | median sample err %.3f deg\n', ...
            Ns, bestASBS, bestRandom, bestASBSRotErrDeg, medianSampleRotErrDeg);
    end

    T = vertcat(rows{:});
end

function errs = rotationErrorDegBatch(q_batch, Rtrue)
    B = size(q_batch, 2);
    errs = zeros(1, B);
    for i = 1:B
        errs(i) = rotationErrorDeg(quatToRotm(q_batch(:, i)), Rtrue);
    end
end

function plot_asbs_experiment_results(T)
    % Means over repeats
    G = groupsummary(T, {'beta', 'outlierRatio', 'Ns'}, 'mean', ...
        {'bestASBS_TLS', 'bestRandom_TLS', 'bestASBS_rotErrDeg', ...
         'medianSample_rotErrDeg', 'fracSamplesBelow5Deg'});

    % 1. beta effect at max Ns
    maxNs = max(T.Ns);
    figure('Color', 'w'); hold on; box on; grid on;
    outVals = unique(G.outlierRatio)';
    for outlierRatio = outVals
        idx = G.Ns == maxNs & abs(G.outlierRatio - outlierRatio) < 1e-12;
        plot(G.beta(idx), G.mean_bestASBS_TLS(idx), '-o', ...
            'DisplayName', sprintf('outliers = %.2f', outlierRatio));
    end
    xlabel('\beta');
    ylabel('Best ASBS TLS');
    title(sprintf('Effect of beta at Ns = %d', maxNs));
    legend('Location', 'best');

    % 2. Ns effect at the middle beta
    betaVals = unique(T.beta);
    beta0 = betaVals(ceil(numel(betaVals) / 2));
    figure('Color', 'w'); hold on; box on; grid on;
    for outlierRatio = outVals
        idx = abs(G.beta - beta0) < 1e-12 & abs(G.outlierRatio - outlierRatio) < 1e-12;
        semilogx(G.Ns(idx), G.mean_bestASBS_TLS(idx), '-o', ...
            'DisplayName', sprintf('outliers = %.2f', outlierRatio));
    end
    xlabel('N_s');
    ylabel('Best ASBS TLS');
    title(sprintf('Effect of sample count at beta = %.2f', beta0));
    legend('Location', 'best');

    % 3. Concentration
    figure('Color', 'w'); hold on; box on; grid on;
    for outlierRatio = outVals
        idx = G.Ns == maxNs & abs(G.outlierRatio - outlierRatio) < 1e-12;
        plot(G.beta(idx), G.mean_medianSample_rotErrDeg(idx), '-o', ...
            'DisplayName', sprintf('outliers = %.2f', outlierRatio));
    end
    xlabel('\beta');
    ylabel('Median sample rotation error [deg]');
    title(sprintf('Sample concentration around true solution at Ns = %d', maxNs));
    legend('Location', 'best');

    % 4. Outlier robustness
    figure('Color', 'w'); hold on; box on; grid on;
    idx = G.Ns == maxNs & abs(G.beta - beta0) < 1e-12;
    plot(G.outlierRatio(idx), G.mean_bestASBS_rotErrDeg(idx), '-o');
    xlabel('Outlier ratio');
    ylabel('Best ASBS rotation error [deg]');
    title(sprintf('Outlier robustness at beta = %.2f, Ns = %d', beta0, maxNs));
end

function make_wahba_two_panel_figure(params, q_samples, q_refined)
    % (a) TLS energy on a geodesic slice of S^3; (b) inlier correspondences on S^2

    if nargin < 2
        q_samples = [];
    end
    if nargin < 3 || isempty(q_refined)
        if isfield(params, 'qtrue')
            q_refined = params.qtrue;
        else
            q_refined = [1;0;0;0];
        end
    end

    % Centre at the true quaternion if available
    if isfield(params, 'qtrue')
        q_center = params.qtrue(:);
    else
        q_center = q_refined(:);
    end
    q_center = q_center / norm(q_center);
    if q_center(1) < 0
        q_center = -q_center;
    end

    fig = figure('Color', 'w', 'Position', [80 80 1450 570]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % (a) Geodesic slice of S^3
    nexttile(1);
    ng = 151;
    rmax = 1.20; % radians on S^3
    [xg, yg] = meshgrid(linspace(-rmax, rmax, ng), linspace(-rmax, rmax, ng));
    U = null(q_center'); % tangent basis at q_center
    u1 = U(:,1);
    u2 = U(:,2);
    Q = zeros(4, numel(xg));
    for k = 1:numel(xg)
        v = xg(k) * u1 + yg(k) * u2;
        r = norm(v);
        if r < 1e-12
            q = q_center;
        else
            q = cos(r) * q_center + sin(r) * v / r;
        end
        if q(1) < 0
            q = -q;
        end
        Q(:,k) = q;
    end
    F = wahba_tls_energy_local(Q, params, false);
    Fimg = reshape(F, size(xg));

    contourf(xg, yg, Fimg, 34, 'LineStyle', 'none');
    axis equal tight;
    box on;
    xlabel('$\xi_1$ on $T_{q_\star}S^3$', 'Interpreter', 'latex', 'FontSize', 24);
    ylabel('$\xi_2$ on $T_{q_\star}S^3$', 'Interpreter', 'latex', 'FontSize', 24);
    title('(a) TLS energy on a geodesic slice of $S^3$', 'Interpreter', 'latex', 'FontSize', 24);
    cb = colorbar;
    cb.Label.String = '$J(q)$';
    cb.Label.Interpreter = 'latex';
    hold on;
    plot(0, 0, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 14, 'DisplayName', '$q_\star$');

    % Samples projected on the slice
    if ~isempty(q_samples)
        ns = min(size(q_samples, 2), 700);
        idx = round(linspace(1, size(q_samples, 2), ns));
        coords = project_to_tangent_slice(q_samples(:, idx), q_center, u1, u2);
        keep = abs(coords(1,:)) <= rmax & abs(coords(2,:)) <= rmax;
        plot(coords(1, keep), coords(2, keep), '.', 'Color', [0 0 0], ...
            'MarkerSize', 4, 'DisplayName', 'ASBS-M samples');
    end

    % (b) Correspondences on S^2
    nexttile(2);
    hold on;
    axis equal;
    box on;
    view(38, 22);

    [XS, YS, ZS] = sphere(80);
    surf(XS, YS, ZS, 'FaceColor', [0.94 0.94 0.94], 'EdgeColor', 'none', ...
        'FaceAlpha', 0.26, 'HandleVisibility', 'off');
    light('Position', [3 2 4]);
    lighting gouraud;

    if isfield(params, 'Rtrue')
        Rtrue = params.Rtrue;
    elseif isfield(params, 'qtrue')
        Rtrue = quatToRotm_local(params.qtrue(:));
    else
        Rtrue = quatToRotm_local(q_center);
    end
    [axisVec, angleTrue] = axis_angle_from_rotm(Rtrue);
    if norm(axisVec) < 1e-12
        axisVec = [0; 0; 1];
    end

    % Two true inliers
    if isfield(params, 'inlierMask')
        inliers = find(params.inlierMask);
    else
        inliers = 1:size(params.A, 2);
    end

    if numel(inliers) < 2
        inliers = 1:min(2, size(params.A, 2));
    end

    sel = inliers(1:min(2, numel(inliers)));

    fig = gcf;
    fig.Color = 'w';
    fig.Units = 'centimeters';
    fig.Position = [4 4 9.0 8.2];

    ax = gca;
    hold(ax, 'on');

    set(ax, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 15, ...
        'LineWidth', 1.15, ...
        'TickLabelInterpreter', 'latex', ...
        'GridAlpha', 0.22, ...
        'MinorGridAlpha', 0.12, ...
        'GridLineStyle', '-', ...
        'MinorGridLineStyle', ':', ...
        'Box', 'on', ...
        'Layer', 'top');

    grid(ax, 'on');
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.ZMinorGrid = 'on';

    axis(ax, 'equal');
    axis(ax, 'vis3d');

    xlim(ax, [-1.05 1.05]);
    ylim(ax, [-1.05 1.05]);
    zlim(ax, [-1.05 1.05]);

    view(ax, [38 24]);

    colors = [
        0.000 0.447 0.741;
        0.850 0.325 0.098
    ];

    axisColor = [0.08 0.08 0.08];

    plot3(ax, ...
        [-axisVec(1) axisVec(1)], ...
        [-axisVec(2) axisVec(2)], ...
        [-axisVec(3) axisVec(3)], ...
        '-', ...
        'Color', axisColor, ...
        'LineWidth', 2.8);

    quiver3(ax, ...
        0, 0, 0, ...
        axisVec(1), axisVec(2), axisVec(3), ...
        0, ...
        'Color', axisColor, ...
        'LineWidth', 2.8, ...
        'MaxHeadSize', 0.45);

    for jj = 1:numel(sel)

        i = sel(jj);

        a = params.A(:, i);
        b_exact = Rtrue * a;
        b_meas = params.Bmeas(:, i);

        c = colors(jj, :);

        % a_i
        quiver3(ax, ...
            0, 0, 0, ...
            a(1), a(2), a(3), ...
            0, ...
            'Color', c, ...
            'LineWidth', 2.4, ...
            'MaxHeadSize', 0.22);

        plot3(ax, ...
            a(1), a(2), a(3), ...
            'o', ...
            'Color', c, ...
            'MarkerFaceColor', c, ...
            'MarkerSize', 8.5, ...
            'LineWidth', 1.4);

        % R_true a_i
        hq = quiver3(ax, ...
            0, 0, 0, ...
            b_exact(1), b_exact(2), b_exact(3), ...
            0, ...
            'Color', c, ...
            'LineWidth', 2.3, ...
            'MaxHeadSize', 0.22);

        hq.LineStyle = '--';

        plot3(ax, ...
            b_exact(1), b_exact(2), b_exact(3), ...
            's', ...
            'Color', c, ...
            'MarkerFaceColor', 'w', ...
            'MarkerSize', 9.0, ...
            'LineWidth', 1.6);

        % Measured endpoint
        plot3(ax, ...
            b_meas(1), b_meas(2), b_meas(3), ...
            '^', ...
            'Color', c, ...
            'MarkerFaceColor', c, ...
            'MarkerSize', 8.5, ...
            'LineWidth', 1.4);

        % Rotation arc
        tt = linspace(0, 1, 120);
        arc = zeros(3, numel(tt));

        for k = 1:numel(tt)
            Rk = axis_angle_to_rotm(axisVec, angleTrue * tt(k));
            arc(:, k) = Rk * a;
        end

        plot3(ax, ...
            arc(1, :), arc(2, :), arc(3, :), ...
            '-', ...
            'Color', c, ...
            'LineWidth', 2.1);
    end

    xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel(ax, '$y$', 'Interpreter', 'latex', 'FontSize', 18);
    zlabel(ax, '$z$', 'Interpreter', 'latex', 'FontSize', 18);

    xticks(ax, -1:0.5:1);
    yticks(ax, -1:0.5:1);
    zticks(ax, -1:0.5:1);
end

function coords = project_to_tangent_slice(Q, q0, u1, u2)
    Q = Q ./ vecnorm(Q);
    for k = 1:size(Q,2)
        if dot(Q(:,k), q0) < 0
            Q(:,k) = -Q(:,k);
        end
    end
    c = max(min(q0' * Q, 1), -1);
    theta = acos(c);
    coords = zeros(2, size(Q,2));
    for k = 1:size(Q,2)
        if theta(k) < 1e-12
            v = zeros(4,1);
        else
            v = theta(k) / sin(theta(k)) * (Q(:,k) - cos(theta(k)) * q0);
        end
        coords(:,k) = [dot(u1, v); dot(u2, v)];
    end
end

function F = wahba_tls_energy_local(q, params, smoothFlag)
    q = q ./ vecnorm(q);
    B = size(q, 2);
    F = zeros(1, B);
    c2 = params.cbar^2;
    tau = params.tlsSmoothing;
    for i = 1:params.Ncorr
        ra = quatRotateBatch_local(q, params.A(:, i));
        r = ra - params.Bmeas(:, i);
        s = sum(r.^2, 1) / (params.sigmas(i)^2);
        if smoothFlag
            m = min(s, c2);
            rho = m - tau * log(exp(-(s - m) / tau) + exp(-(c2 - m) / tau));
        else
            rho = min(s, c2);
        end
        F = F + rho;
    end
end

function y = quatRotateBatch_local(q, a)
    a1 = a(1); a2 = a(2); a3 = a(3);
    qw = q(1, :);
    qx = q(2, :); qy = q(3, :); qz = q(4, :);
    tx = 2 * (qy * a3 - qz * a2);
    ty = 2 * (qz * a1 - qx * a3);
    tz = 2 * (qx * a2 - qy * a1);
    y1 = a1 + qw .* tx + (qy .* tz - qz .* ty);
    y2 = a2 + qw .* ty + (qz .* tx - qx .* tz);
    y3 = a3 + qw .* tz + (qx .* ty - qy .* tx);
    y = [y1; y2; y3];
end

function R = quatToRotm_local(q)
    q = q(:) / norm(q);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1-2*(y^2+z^2), 2*(x*y-w*z),   2*(x*z+w*y); ...
         2*(x*y+w*z),   1-2*(x^2+z^2), 2*(y*z-w*x); ...
         2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x^2+y^2)];
end

function [axisVec, angleVal] = axis_angle_from_rotm(R)
    val = (trace(R) - 1) / 2;
    val = max(min(val, 1), -1);
    angleVal = acos(val);
    if angleVal < 1e-12
        axisVec = [0;0;1];
    else
        axisVec = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)] / (2*sin(angleVal));
        axisVec = axisVec / max(norm(axisVec), eps);
    end
end

function R = axis_angle_to_rotm(axisVec, angleVal)
    axisVec = axisVec(:) / max(norm(axisVec), eps);
    x = axisVec(1); y = axisVec(2); z = axisVec(3);
    K = [0 -z y; z 0 -x; -y x 0];
    R = eye(3) + sin(angleVal) * K + (1 - cos(angleVal)) * (K * K);
end

function R = axisAngleToRotm(axis, angle)
    axis = axis(:) / max(norm(axis), eps);
    x = axis(1); y = axis(2); z = axis(3);
    K = [0 -z y; z 0 -x; -y x 0];
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * (K * K);
end

function coords = rotmToAxisAngle(R)
    angle = acos(max(-1, min(1, (trace(R) - 1) / 2)));
    if angle < 1e-12
        axis = [0; 0; 0];
    else
        axis = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)] / (2*sin(angle));
    end
    coords = angle * axis;
end

function params = makeSyntheticWahba(params)
    axis = params.trueAxis(:) / norm(params.trueAxis);
    angle = deg2rad(params.trueAngleDeg);
    qtrue = [cos(angle/2); axis * sin(angle/2)];
    Rtrue = quatToRotm(qtrue);

    A = normalizeColumns(randn(3, params.Ncorr));
    Bclean = Rtrue * A;

    nOut = round(params.outlierRatio * params.Ncorr);
    outIdx = false(1, params.Ncorr);
    p = randperm(params.Ncorr, nOut);
    outIdx(p) = true;
    inlierMask = ~outIdx;

    Bmeas = zeros(3, params.Ncorr);
    for i = 1:params.Ncorr
        if inlierMask(i)
            Bmeas(:, i) = Bclean(:, i) + params.inlierNoise * randn(3, 1);
            Bmeas(:, i) = Bmeas(:, i) / norm(Bmeas(:, i));
        else
            Bmeas(:, i) = randn(3, 1);
            Bmeas(:, i) = Bmeas(:, i) / norm(Bmeas(:, i));
        end
    end

    params.A = A;
    params.Bmeas = Bmeas;
    params.sigmas = params.sigmaMeas * ones(1, params.Ncorr);
    params.inlierMask = inlierMask;
    params.Rtrue = Rtrue;
    params.qtrue = qtrue;
end

function q = uniformQuaternions(B)
    q = randn(4, B);
    q = q ./ vecnorm(q);
end

function X = normalizeColumns(X)
    X = X ./ max(vecnorm(X), eps);
end

function q = canonicalQuat(q)
    q = q ./ vecnorm(q);
    flip = q(1, :) < 0;
    q(:, flip) = -q(:, flip);
end

function Pv = batchProjector(q, v, params)
    if isdlarray(v) && ~isdlarray(q)
        q = dlarray(q, 'CB');
    end
    dots = sum(q .* v, 1);
    Pv = v - q .* dots;
end

function q_new = Retraction(q, params)
    n = sqrt(sum(q.^2, 1));
    q_new = q ./ max(n, eps);
end

function q_traj = simulate_trajectories(net, q0, params)
    B = size(q0, 2);
    q_traj = zeros(params.d, B, params.N_steps + 1);
    q_traj(:, :, 1) = Retraction(q0, params);

    for j = 1:params.N_steps
        t = (j - 1) * params.dt;
        q_curr = q_traj(:, :, j);
        u_val = predict_net(net, q_curr, params.d, B, true, t);
        noise = randn(params.d, B);

        if isfield(params, 'sig'), s = params.sig(j); else, s = params.sigma; end
        drift = s * batchProjector(q_curr, u_val, params) * params.dt;
        diffTerm = s * sqrt(params.dt) * batchProjector(q_curr, noise, params);
        q_traj(:, :, j + 1) = Retraction(q_curr + drift + diffTerm, params);
    end
end

function out = predict_net(net, X, d, B, hasTime, t)
    if hasTime
        inputs = dlarray([X; t * ones(1, B)], 'CB');
    else
        inputs = dlarray(X, 'CB');
    end
    out = extractdata(predict(net, inputs));
end

function [gradE, E_val] = compute_energy_grad(q_batch, params)
    q_dl = dlarray(q_batch, 'CB');
    [E_val, gradE] = dlfeval(@energy_and_grad, q_dl, params);
    gradE = extractdata(gradE);
    E_val = extractdata(E_val);
end

function [E_val, gradE] = energy_and_grad(q_dl, params)
    qn = q_dl ./ sqrt(sum(q_dl.^2, 1));
    F = wahba_tls_energy_dl(qn, params);
    E_val = params.beta * F;
    gradE = dlgradient(sum(E_val), q_dl);
end

function F = wahba_tls_energy_dl(q, params)
    B = size(q, 2);
    F = q(1, :) * 0;
    c2 = params.cbar^2;
    tau = params.tlsSmoothing;

    for i = 1:params.Ncorr
        ra = quatRotateBatch(q, params.A(:, i));
        r = ra - params.Bmeas(:, i);
        s = sum(r.^2, 1) / (params.sigmas(i)^2);
        if params.useSmoothTLS
            m = min(s, c2);
            rho = m - tau * log(exp(-(s - m) / tau) + exp(-(c2 - m) / tau));
        else
            rho = min(s, c2);
        end
        F = F + rho;
    end
end

function F = wahba_tls_energy(q, params, smoothFlag)
    q = Retraction(q, params);
    B = size(q, 2);
    F = zeros(1, B);
    c2 = params.cbar^2;
    tau = params.tlsSmoothing;
    for i = 1:params.Ncorr
        ra = quatRotateBatch(q, params.A(:, i));
        r = ra - params.Bmeas(:, i);
        s = sum(r.^2, 1) / (params.sigmas(i)^2);
        if smoothFlag
            m = min(s, c2);
            rho = m - tau * log(exp(-(s - m) / tau) + exp(-(c2 - m) / tau));
        else
            rho = min(s, c2);
        end
        F = F + rho;
    end
end

function info = wahba_residuals(q, params)
    q = Retraction(q, params);
    S = zeros(1, params.Ncorr);
    raw = zeros(1, params.Ncorr);
    for i = 1:params.Ncorr
        ra = quatRotateBatch(q, params.A(:, i));
        r = ra - params.Bmeas(:, i);
        raw(i) = norm(r);
        S(i) = raw(i)^2 / params.sigmas(i)^2;
    end
    info.raw = raw;
    info.s = S;
    info.rho = min(S, params.cbar^2);
end

function y = quatRotateBatch(q, a)
    % Rotate a by each scalar-first quaternion (q: 4 x B)
    a1 = a(1); a2 = a(2); a3 = a(3);
    qw = q(1, :);
    qx = q(2, :); qy = q(3, :); qz = q(4, :);

    % t = 2 cross(q_vec, a)
    tx = 2 * (qy * a3 - qz * a2);
    ty = 2 * (qz * a1 - qx * a3);
    tz = 2 * (qx * a2 - qy * a1);

    % y = a + qw t + cross(q_vec, t)
    y1 = a1 + qw .* tx + (qy .* tz - qz .* ty);
    y2 = a2 + qw .* ty + (qz .* tx - qx .* tz);
    y3 = a3 + qw .* tz + (qx .* ty - qy .* tx);
    y = [y1; y2; y3];
end

function [gradients, loss] = computeLossU(netU, X_batch, t_batch, V_target, sigma, BatchSize, params)
    inputs = dlarray([X_batch; t_batch], 'CB');
    U_pred = forward(netU, inputs);
    U_proj = batchProjector(X_batch, U_pred, params);
    V_dl = dlarray(V_target, 'CB');
    residuals = U_proj + sigma .* V_dl;
    if isfield(params, 'loss_weight') && params.loss_weight
        loss = sum((residuals ./ sigma).^2, 'all') / BatchSize;  % 1/sigma_t^2 weight
    else
        loss = sum(residuals.^2, 'all') / BatchSize;
    end
    gradients = dlgradient(loss, netU.Learnables);
end

function [gradients, loss] = computeLossH(netH, X_N, b_target, d, BatchSize, params)
    inputs = dlarray(X_N, 'CB');
    H_pred = forward(netH, inputs);
    H_proj = batchProjector(X_N, H_pred, params);
    b_dl = dlarray(b_target, 'CB');
    residuals = H_proj - b_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
    gradients = dlgradient(loss, netH.Learnables);
end

function gradients = clipGradients(gradients, clipNorm)
    if isempty(clipNorm) || ~isfinite(clipNorm) || clipNorm <= 0
        return;
    end
    totalSq = 0;
    for i = 1:size(gradients, 1)
        g = gradients.Value{i};
        if ~isempty(g)
            totalSq = totalSq + double(gather(extractdata(sum(g.^2, 'all'))));
        end
    end
    nrm = sqrt(totalSq);
    if nrm > clipNorm
        scale = clipNorm / (nrm + 1e-12);
        for i = 1:size(gradients, 1)
            if ~isempty(gradients.Value{i})
                gradients.Value{i} = gradients.Value{i} * scale;
            end
        end
    end
end

function R = quatToRotm(q)
    q = q(:) / norm(q);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1 - 2*(y^2 + z^2), 2*(x*y - z*w),     2*(x*z + y*w);
         2*(x*y + z*w),     1 - 2*(x^2 + z^2), 2*(y*z - x*w);
         2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x^2 + y^2)];
end

function q = rotmToQuat(R)
    tr = trace(R);
    if tr > 0
        S = sqrt(tr + 1.0) * 2;
        w = 0.25 * S;
        x = (R(3,2) - R(2,3)) / S;
        y = (R(1,3) - R(3,1)) / S;
        z = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;
        w = (R(3,2) - R(2,3)) / S;
        x = 0.25 * S;
        y = (R(1,2) + R(2,1)) / S;
        z = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;
        w = (R(1,3) - R(3,1)) / S;
        x = (R(1,2) + R(2,1)) / S;
        y = 0.25 * S;
        z = (R(2,3) + R(3,2)) / S;
    else
        S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;
        w = (R(2,1) - R(1,2)) / S;
        x = (R(1,3) + R(3,1)) / S;
        y = (R(2,3) + R(3,2)) / S;
        z = 0.25 * S;
    end
    q = [w; x; y; z];
    q = q / norm(q);
end

function err = rotationErrorDeg(Rhat, Rtrue)
    c = (trace(Rhat' * Rtrue) - 1) / 2;
    c = min(1, max(-1, c));
    err = rad2deg(acos(c));
end

%% Method comparison: R-ASBS, Riemannian MALA, multistart RGD, exact target

function [Tc, store] = run_method_comparison(params, NsList, q_asbs, q_unif, rep, runSeed, cmpCfg, costASBS)
    % Same instance and metrics for every method: concentration around q* and best-of-n selection
    maxNs  = max(NsList);
    W      = wahba_quadforms(params);                  % 16 x N quadratic forms / sigma_i^2
    c2     = params.cbar^2;
    smooth = params.useSmoothTLS;
    efun   = @(X) wahba_energy_grad(X, W, c2, params.tlsSmoothing, params.beta, smooth);   % ASBS-M target
    tls    = @(X) wahba_energy_grad(X, W, c2, params.tlsSmoothing, 1, false);              % exact TLS
    hasTruth = isfield(params, 'qtrue');
    if hasTruth, qtrue = params.qtrue(:); else, qtrue = []; end

    % Samplers
    S = struct('name', {}, 'Q', {}, 'trainTime', {}, 'sampleTime', {}, 'trainEvals', {}, 'budget', {}, 'acc', {});
    S(end+1) = struct('name', 'ASBS-M', 'Q', q_asbs, 'trainTime', costASBS.trainTime, ...
        'sampleTime', costASBS.sampleTime, 'trainEvals', costASBS.trainEvals, 'budget', NaN, 'acc', NaN);

    if cmpCfg.runRASBS
        fprintf('\n[cmp] Training R-ASBS (sphere-exact) with the same networks / epochs / batch / optimiser / sigma(t) as ASBS-M...\n');
        paramsR = params;
        t0 = tic;
        netR = train_rasbs_sphere(paramsR, cmpCfg);
        tTrain = toc(t0);
        t0 = tic;
        trajR = simulate_trajectories(netR, uniformQuaternions(maxNs), paramsR);
        S(end+1) = struct('name', 'R-ASBS', 'Q', trajR(:, :, end), 'trainTime', tTrain, ...
            'sampleTime', toc(t0), 'trainEvals', params.K_epochs * params.B, 'budget', NaN, 'acc', NaN);
        clear trajR
    end

    if cmpCfg.runMALA
        budget  = params.K_epochs * params.B;           % = ASBS-M training budget
        stepsCk = max(1, round(cmpCfg.mala_budget_mult * budget / maxNs));
        fprintf('[cmp] Riemannian MALA: %d chains, checkpoints at %s steps\n', maxNs, mat2str(stepsCk));
        [ck, infoM] = mala_sphere(uniformQuaternions(maxNs), efun, stepsCk, cmpCfg.mala_delta0, cmpCfg.mala_target_acc);
        for c = 1:numel(stepsCk)
            S(end+1) = struct('name', sprintf('MALA x%g', cmpCfg.mala_budget_mult(c)), 'Q', ck{c}, ...
                'trainTime', 0, 'sampleTime', infoM.time_ck(c), 'trainEvals', stepsCk(c) * maxNs, ...
                'budget', cmpCfg.mala_budget_mult(c), 'acc', infoM.acc_ck(c)); %#ok<AGROW>
        end
    end

    S(end+1) = struct('name', 'Uniform', 'Q', q_unif, 'trainTime', 0, 'sampleTime', 0, ...
        'trainEvals', 0, 'budget', NaN, 'acc', NaN);

    % Multistart RGD on the uniform pool
    rgd = [];
    if cmpCfg.runRGD
        fprintf('[cmp] Multistart RGD, K = %s\n', mat2str(cmpCfg.rgd_K));
        rgd = run_multistart_rgd(q_unif, NsList, cmpCfg.rgd_K, W, c2);
    end

    % Exact TLS of every pool
    for m = 1:numel(S)
        S(m).F = tls(S(m).Q);
    end

    % q*: best TLS minimiser found by any route
    cand = [];
    if hasTruth, cand = qtrue; end
    for m = 1:numel(S)
        [~, ib] = min(S(m).F);
        cand = [cand, S(m).Q(:, ib)]; %#ok<AGROW>
    end
    [qc, Fc] = rgd_refine_batch(cand, W, c2);
    if ~isempty(rgd), qc = [qc, rgd.Qall]; Fc = [Fc, rgd.Fall]; end
    [Fstar, ib] = min(Fc);
    qstar = qc(:, ib);
    far = rot_err_deg_q(qc, qstar) > 2;                    % other local minima
    if any(far), dF2 = min(Fc(far)) - Fstar; else, dF2 = NaN; end
    errStarTruth = NaN;
    if hasTruth, errStarTruth = rot_err_deg_q(qstar, qtrue); end
    fprintf('[cmp] q*: TLS %.3f, %.2f deg from the truth; beta*(F2 - F*) = %.1f\n', Fstar, errStarTruth, params.beta * dF2);

    % Exact target: Metropolis-adjusted chains started at q*
    errT = [];  pT = nan(1, numel(cmpCfg.deltas_deg));  accT = NaN;
    if cmpCfg.runTarget
        [ckT, infoT] = mala_sphere(repmat(qstar, 1, cmpCfg.ref_chains), efun, cmpCfg.ref_steps, ...
            cmpCfg.mala_delta0, cmpCfg.mala_target_acc);
        errT = rot_err_deg_q(ckT{1}, qstar);
        pT   = arrayfun(@(d) mean(errT < d), cmpCfg.deltas_deg);
        accT = infoT.acc_ck(1);
    end
    i5 = find(cmpCfg.deltas_deg == cmpCfg.succ_deg, 1);
    if isempty(i5), i5 = 1; end
    pUnif = (deg2rad(cmpCfg.deltas_deg) - sin(deg2rad(cmpCfg.deltas_deg))) / pi;   % Haar law of the angle

    % Rows
    base = struct('beta', params.beta, 'outlierRatio', params.outlierRatio, 'repeat', rep, 'seed', runSeed);
    rows = {};
    for m = 1:numel(S)
        errStar = rot_err_deg_q(S(m).Q, qstar);
        if hasTruth, errTrue = rot_err_deg_q(S(m).Q, qtrue); else, errTrue = nan(1, maxNs); end
        for Ns = NsList
            r = empty_cmp_row(base, cmpCfg);
            r.method = S(m).name;  r.Ns = Ns;  r.budgetMult = S(m).budget;  r.acceptRate = S(m).acc;
            F = S(m).F(1:Ns);  eS = errStar(1:Ns);  eT = errTrue(1:Ns);
            [r.bestTLS, ib] = min(F);
            r.gapPct = 100 * (r.bestTLS - Fstar) / Fstar;
            r.bestErrTrue_deg = eT(ib);
            r.success = double(eT(ib) < cmpCfg.succ_deg);
            p = arrayfun(@(d) mean(eS < d), cmpCfg.deltas_deg);
            for k = 1:numel(p), r.(delta_name(cmpCfg.deltas_deg(k))) = p(k); end
            r.pUnif = pUnif(i5);
            r.enrichment = p(i5) / pUnif(i5);
            r.n95 = batch_for_success(p(i5), 0.95);
            r.medianErrStar_deg = median(eS);
            if ~isempty(errT)
                r.KS_target  = ks2(eS, errT);
                r.massRatio  = p(i5) / max(pT(i5), eps);
            end
            Ps = best_of_n_success(F, eT < cmpCfg.succ_deg, cmpCfg.n_curve);
            for k = 1:numel(Ps), r.(sprintf('Psucc_n%d', cmpCfg.n_curve(k))) = Ps(k); end
            r.trainTime_s  = S(m).trainTime;
            r.sampleTime_s = S(m).sampleTime * Ns / maxNs;
            r.energyEvals  = S(m).trainEvals + Ns;         % + Ns TLS evaluations
            r.betaDeltaF2  = params.beta * dF2;
            r.pTarget = pT(i5);
            if ~isempty(errT), r.medianErrTarget_deg = median(errT); end
            rows{end+1} = r; %#ok<AGROW>
        end
    end
    if ~isempty(rgd)
        for j = 1:numel(NsList)
            for k = 1:numel(cmpCfg.rgd_K)
                r = empty_cmp_row(base, cmpCfg);
                r.method = 'Multistart RGD';  r.Ns = NsList(j);  r.K = cmpCfg.rgd_K(k);
                r.bestTLS = rgd.bestF(j, k);
                r.gapPct  = 100 * (r.bestTLS - Fstar) / Fstar;
                if hasTruth
                    r.bestErrTrue_deg = rot_err_deg_q(rgd.bestQ{j, k}, qtrue);
                    r.success = double(r.bestErrTrue_deg < cmpCfg.succ_deg);
                end
                r.sampleTime_s = rgd.time(j, k);
                r.energyEvals  = rgd.evals(j, k);
                r.betaDeltaF2  = params.beta * dF2;
                rows{end+1} = r; %#ok<AGROW>
            end
        end
    end
    if ~isempty(errT)
        r = empty_cmp_row(base, cmpCfg);
        r.method = 'Target (exact, local)';  r.Ns = numel(errT);
        for k = 1:numel(pT), r.(delta_name(cmpCfg.deltas_deg(k))) = pT(k); end
        r.pUnif = pUnif(i5);  r.enrichment = pT(i5) / pUnif(i5);  r.n95 = batch_for_success(pT(i5), 0.95);
        r.medianErrStar_deg = median(errT);  r.pTarget = pT(i5);  r.medianErrTarget_deg = median(errT);
        r.betaDeltaF2 = params.beta * dF2;  r.acceptRate = accT;
        rows{end+1} = r;
    end
    Tc = struct2table([rows{:}]);
    Tc.method = cellstr(Tc.method);

    % Console summary at the largest Ns
    fprintf('\n[cmp] beta %.2f | outliers %.2f | Ns = %d | %d-deg success w.r.t. the truth; p = fraction within %g deg of q*\n', ...
        params.beta, params.outlierRatio, maxNs, cmpCfg.succ_deg, cmpCfg.deltas_deg(i5));
    fprintf('  %-22s %8s %8s %10s %10s %9s %8s %12s\n', 'method', 'err[deg]', 'success', 'p', 'n95', 'median[d]', 'KS', 'grad-E evals');
    for i = 1:height(Tc)
        if Tc.Ns(i) ~= maxNs && ~strcmp(Tc.method{i}, 'Target (exact, local)'), continue; end
        nm = Tc.method{i};
        if strcmp(nm, 'Multistart RGD'), nm = sprintf('RGD (K = %d)', Tc.K(i)); end
        fprintf('  %-22s %8.3f %8.0f %10.2e %10.3g %9.2f %8.3f %12.3g\n', nm, Tc.bestErrTrue_deg(i), Tc.success(i), ...
            Tc.(delta_name(cmpCfg.deltas_deg(i5)))(i), Tc.n95(i), Tc.medianErrStar_deg(i), Tc.KS_target(i), Tc.energyEvals(i));
    end
    fprintf('  uniform: p = %.2e, n95 = %.3g\n', pUnif(i5), batch_for_success(pUnif(i5), 0.95));

    % Error samples for the plots
    store.beta = params.beta;  store.outlierRatio = params.outlierRatio;  store.repeat = rep;
    store.names = {S.name};
    store.errStar = cellfun(@(Q) single(rot_err_deg_q(Q, qstar)), {S.Q}, 'UniformOutput', false);
    store.errTarget = single(errT);
end

function r = empty_cmp_row(base, cmpCfg)
    r = base;
    r.method = '';  r.Ns = NaN;  r.K = NaN;  r.budgetMult = NaN;
    r.bestTLS = NaN;  r.gapPct = NaN;  r.bestErrTrue_deg = NaN;  r.success = NaN;
    for d = cmpCfg.deltas_deg, r.(delta_name(d)) = NaN; end
    r.pUnif = NaN;  r.enrichment = NaN;  r.n95 = NaN;  r.medianErrStar_deg = NaN;
    r.KS_target = NaN;  r.massRatio = NaN;
    for n = cmpCfg.n_curve, r.(sprintf('Psucc_n%d', n)) = NaN; end
    r.trainTime_s = NaN;  r.sampleTime_s = NaN;  r.energyEvals = NaN;
    r.betaDeltaF2 = NaN;  r.pTarget = NaN;  r.medianErrTarget_deg = NaN;  r.acceptRate = NaN;
end

function s = delta_name(d)
    s = strrep(sprintf('p_%gdeg', d), '.', 'p');
end

function n = batch_for_success(p, level)
    % Smallest batch with P(hit) >= level (i.i.d. samples)
    if p <= 0, n = Inf; elseif p >= 1, n = 1; else, n = ceil(log(1 - level) / log(1 - p)); end
end

function P = best_of_n_success(F, ok, n_list)
    % Exact P(min-F sample of a random n-subset satisfies ok)
    [~, o] = sort(F(:), 'ascend');  h = double(ok(:));  h = h(o);  M = numel(h);  k = (1:M)';
    P = nan(size(n_list));
    for i = 1:numel(n_list)
        n = n_list(i);
        if n > M || n < 1, continue; end
        v  = (M - k) >= (n - 1);
        kv = k(v);
        lp = gammaln(M - kv + 1) - gammaln(n) - gammaln(M - kv - n + 2) ...
           - (gammaln(M + 1) - gammaln(n + 1) - gammaln(M - n + 1));
        P(i) = sum(exp(lp) .* h(v));
    end
end

function W = wahba_quadforms(params)
    % Column i = vec(Q_i)/sigma_i^2, q'Q_i q = ||R(q) a_i - b_i||^2 (scalar-first q)
    N = size(params.A, 2);  W = zeros(16, N);  p = [4 1 2 3];
    for i = 1:N
        a = params.A(:, i);  b = params.Bmeas(:, i);
        TL  = a * b' + b * a' - (a' * b) * eye(3);
        cr  = cross(a, b);
        K   = [TL, cr; cr', a' * b];
        Qsl = (a' * a + b' * b) * eye(4) - 2 * K;
        Qsl = 0.5 * (Qsl + Qsl');
        Qsf = Qsl(p, p);
        W(:, i) = Qsf(:) / params.sigmas(i)^2;
    end
end

function [E, G] = wahba_energy_grad(Q, W, c2, tau, beta, smooth)
    % beta*TLS and its Riemannian gradient on S^3 (batched)
    Q = Q ./ vecnorm(Q);
    M = size(Q, 2);
    E = zeros(1, M);
    if nargout > 1, G = zeros(4, M); end
    chunk = max(1, floor(4e6 / size(W, 2)));
    for i0 = 1:chunk:M
        ii = i0:min(i0 + chunk - 1, M);
        q  = Q(:, ii);
        QQ = reshape(q, 4, 1, []) .* reshape(q, 1, 4, []);
        s  = max(reshape(QQ, 16, [])' * W, 0);
        if smooth
            mn  = min(s, c2);
            rho = mn - tau * log(exp(-(s - mn) / tau) + exp(-(c2 - mn) / tau));
            w   = 1 ./ (1 + exp((s - c2) / tau));
        else
            rho = min(s, c2);
            w   = double(s < c2);
        end
        E(ii) = beta * sum(rho, 2)';
        if nargout > 1
            Ws = reshape((w * W')', 4, 4, []);
            g  = 2 * reshape(pagemtimes(Ws, reshape(q, 4, 1, [])), 4, []);
            G(:, ii) = beta * (g - q .* sum(q .* g, 1));
        end
    end
end

function e = rot_err_deg_q(Q, qref)
    % Rotation angle (deg) to qref, sign-invariant
    Q = Q ./ vecnorm(Q);
    qref = qref(:) / norm(qref);
    e = 2 * acosd(min(abs(qref' * Q), 1));
end

function [Q, cost, steps, nCost, nGrad] = rgd_refine_batch(Q0, W, c2)
    % Batched Riemannian gradient descent on the exact TLS, Armijo backtracking
    max_it = 100; eta = 1.0; c1 = 1e-4; rho = 0.5; max_bt = 35; gtol = 1e-8; ftol = 1e-12;
    Q = Q0 ./ vecnorm(Q0);
    Q(:, Q(1, :) < 0) = -Q(:, Q(1, :) < 0);
    K = size(Q, 2);
    cost = wahba_energy_grad(Q, W, c2, 0, 1, false);
    nCost = ones(1, K);  nGrad = zeros(1, K);  steps = zeros(1, K);
    active = true(1, K);
    for it = 1:max_it
        idx = find(active);
        if isempty(idx), break; end
        [~, g] = wahba_energy_grad(Q(:, idx), W, c2, 0, 1, false);
        nGrad(idx) = nGrad(idx) + 1;
        gns  = sum(g.^2, 1);
        stop = gns <= gtol^2;
        active(idx(stop)) = false;
        idx = idx(~stop);  g = g(:, ~stop);  gns = gns(~stop);
        if isempty(idx), break; end
        step = eta ./ max(sqrt(gns), 1);
        accepted = false(1, numel(idx));
        qnew = Q(:, idx);  cnew = cost(idx);
        pending = true(1, numel(idx));
        for bt = 1:max_bt
            p = find(pending);
            if isempty(p), break; end
            qt = Q(:, idx(p)) - step(p) .* g(:, p);
            qt = qt ./ vecnorm(qt);
            tc = wahba_energy_grad(qt, W, c2, 0, 1, false);
            nCost(idx(p)) = nCost(idx(p)) + 1;
            ok = tc <= cost(idx(p)) - c1 * step(p) .* gns(p);
            qnew(:, p(ok)) = qt(:, ok);  cnew(p(ok)) = tc(ok);
            accepted(p(ok)) = true;  pending(p(ok)) = false;
            step(p(~ok)) = step(p(~ok)) * rho;
        end
        active(idx(~accepted)) = false;
        acc  = idx(accepted);
        prev = cost(acc);
        Q(:, acc) = qnew(:, accepted);  cost(acc) = cnew(accepted);  steps(acc) = steps(acc) + 1;
        active(acc((prev - cost(acc)) ./ max(1, abs(prev)) <= ftol)) = false;
    end
    Q(:, Q(1, :) < 0) = -Q(:, Q(1, :) < 0);
end

function R = run_multistart_rgd(q_unif, NsList, Klist, W, c2)
    % Multistart RGD: rank Ns uniform starts by TLS, refine the top K, keep the best
    t0 = tic;
    F0 = wahba_energy_grad(q_unif, W, c2, 0, 1, false);
    tRank = toc(t0);
    Kmax = max(Klist);
    order = cell(numel(NsList), 1);
    idxAll = [];
    for j = 1:numel(NsList)
        [~, o] = sort(F0(1:NsList(j)), 'ascend');
        order{j} = o(1:min(Kmax, NsList(j)));
        idxAll = union(idxAll, order{j});
    end
    t0 = tic;
    [Qr, Fr, ~, nC, nG] = rgd_refine_batch(q_unif(:, idxAll), W, c2);
    tRef = toc(t0);
    work = nC + nG;
    pos = zeros(1, size(q_unif, 2));  pos(idxAll) = 1:numel(idxAll);
    R.bestF = nan(numel(NsList), numel(Klist));  R.bestQ = cell(numel(NsList), numel(Klist));
    R.time  = nan(numel(NsList), numel(Klist));  R.evals = nan(numel(NsList), numel(Klist));
    for j = 1:numel(NsList)
        for k = 1:numel(Klist)
            sel = pos(order{j}(1:min(Klist(k), NsList(j))));
            [R.bestF(j, k), b] = min(Fr(sel));
            R.bestQ{j, k} = Qr(:, sel(b));
            R.evals(j, k) = NsList(j) + sum(work(sel));                     % ranking + refinement
            R.time(j, k)  = tRank * NsList(j) / numel(F0) + tRef * sum(work(sel)) / sum(work);
        end
    end
    R.Qall = Qr;  R.Fall = Fr;
end

function [X_ck, info] = mala_sphere(X0, efun, steps_ck, delta0, target_acc)
    % Metropolis-adjusted Riemannian Langevin on S^3; step adapted to target_acc
    X = X0 ./ vecnorm(X0);
    C = size(X, 2);
    [E, G] = efun(X);
    log_d = log(delta0);
    n = max(steps_ck);
    X_ck = cell(numel(steps_ck), 1);
    info.acc_ck = nan(numel(steps_ck), 1);  info.delta_ck = nan(numel(steps_ck), 1);  info.time_ck = nan(numel(steps_ck), 1);
    acc_hist = nan(n, 1);
    t0 = tic;
    for k = 1:n
        d = exp(log_d);
        b = -0.5 * G;
        z = randn(size(X));
        z = z - X .* sum(z .* X, 1);
        V = d * b + sqrt(d) * z;
        Y = sphere_expmap(X, V);
        [Ey, Gy] = efun(Y);
        Vb = sphere_logmap(Y, X) - d * (-0.5 * Gy);
        log_a = -(Ey - E) - (sum(Vb.^2, 1) - sum((V - d * b).^2, 1)) / (2 * d);
        log_a(~isfinite(log_a)) = -Inf;
        acc = log(rand(1, C)) < log_a;
        X(:, acc) = Y(:, acc);  E(acc) = Ey(acc);  G(:, acc) = Gy(:, acc);
        acc_hist(k) = mean(exp(min(log_a, 0)));
        log_d = min(log_d + 2 * (acc_hist(k) - target_acc) / k^0.6, log(0.5));
        for c = find(steps_ck == k)
            X_ck{c} = X;
            info.time_ck(c)  = toc(t0);
            info.delta_ck(c) = exp(log_d);
            info.acc_ck(c)   = mean(acc_hist(max(1, k - 49):k));
        end
    end
end

function Y = sphere_expmap(X, V)
    nv = vecnorm(V, 2, 1);
    Y = X .* cos(nv) + V ./ max(nv, 1e-300) .* sin(nv);
    small = nv < 1e-12;
    Y(:, small) = X(:, small) + V(:, small);
    Y = Y ./ vecnorm(Y, 2, 1);
end

function V = sphere_logmap(X, Y)
    c = max(min(sum(X .* Y, 1), 1), -1);
    U = Y - c .* X;
    V = U .* (acos(c) ./ max(vecnorm(U, 2, 1), 1e-300));
end

function netU = train_rasbs_sphere(params, cmpCfg)
    % R-ASBS (Alg. 1), exact S^3 geometry, same networks, optimiser and sigma(t) as ASBS-M.
    % Controller: geodesic bridge + parallel transport; corrector: S^3 heat-kernel score.
    [netU, netH] = initialize_networks(params);
    avgU = []; avgSqU = [];
    avgH = []; avgSqH = [];
    V1 = params.V1;                                              % total reference variance
    tau = [0, cumsum(params.sig(1:end-1)'.^2) * params.dt];      % reference clock tau(t_j)
    nt = cmpCfg.rasbs_nt;
    [iterU, iterH] = deal(0);
    for k = 1:params.K_epochs
        lr_k = params.lr * (params.lr_end + (1 - params.lr_end) * 0.5 * (1 + cos(pi * (k - 1) / params.K_epochs)));
        q0 = uniformQuaternions(params.B);
        q_traj = simulate_trajectories(netU, q0, params);
        q1 = q_traj(:, :, params.N_steps + 1);
        h1 = predict_net(netH, q1, params.d, params.B, false, 0);
        [gradE, ~] = compute_energy_grad(q1, params);
        a1 = batchProjector(q1, gradE + h1, params);

        for it = 1:params.n_inner
            % Bridge at clock fraction tau_j / V1
            j = randi(params.N_steps, 1, params.B * nt);
            t = (j - 1) * params.dt;
            [Xt, At] = sphere_bridge_transport(repmat(q0, 1, nt), repmat(q1, 1, nt), repmat(a1, 1, nt), tau(j) / V1, V1);
            iterU = iterU + 1;
            [gradU, lossU] = dlfeval(@computeLossU, netU, Xt, t, At, reshape(params.sig(j), 1, []), params.B * nt, params);
            gradU = clipGradients(gradU, params.gradClip);
            [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, iterU, lr_k);
        end

        q_resampled = simulate_trajectories(netU, q0, params);
        q1b = q_resampled(:, :, params.N_steps + 1);
        b_target = s3_heat_kernel_score(q0, q1b, V1);
        for it = 1:params.n_inner
            iterH = iterH + 1;
            [gradH, lossH] = dlfeval(@computeLossH, netH, q1b, b_target, params.d, params.B, params);
            gradH = clipGradients(gradH, params.gradClip);
            [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, iterH, lr_k);
        end

        if params.verbose && (mod(k, params.printEvery) == 0 || k == 1)
            fprintf('R-ASBS epoch %04d | Loss U %.4e | Loss H %.4e | best train TLS %.4f\n', k, ...
                double(gather(extractdata(lossU))), double(gather(extractdata(lossH))), ...
                min(wahba_tls_energy(q1b, params, false)));
        end
    end
end

function [Xt, At] = sphere_bridge_transport(X0, X1, A1, t, V1)
    % Brownian-bridge point on S^n (clock fraction t); A1 parallel-transported to Xt
    [d, M] = size(X0);
    c  = max(min(sum(X0 .* X1, 1), 1), -1);
    th = acos(c);
    u  = X1 - X0 .* c;
    un = vecnorm(u, 2, 1);
    u  = u ./ max(un, 1e-300);  u(:, un < 1e-12) = 0;
    mu = X0 .* cos(t .* th) + u .* sin(t .* th);
    z  = randn(d, M);
    z  = z - mu .* sum(z .* mu, 1);
    Xt = mu + z .* sqrt(V1 * t .* (1 - t));
    Xt = Xt ./ vecnorm(Xt, 2, 1);
    c1 = max(min(sum(X1 .* Xt, 1), 1), -1);
    ph = acos(c1);
    e  = Xt - X1 .* c1;
    en = vecnorm(e, 2, 1);
    e  = e ./ max(en, 1e-300);  e(:, en < 1e-12) = 0;
    ap = sum(A1 .* e, 1);
    At = A1 - e .* ap + ap .* (e .* cos(ph) - X1 .* sin(ph));
end

function b = s3_heat_kernel_score(X0, X1, V)
    % grad_{x1} log p_V(x1 | x0), exact heat kernel of S^3 (winding terms included)
    c  = max(min(sum(X0 .* X1, 1), 1), -1);
    th = min(acos(c), pi - 0.02);
    u  = X0 - X1 .* c;                                   % tangent at x1 towards x0
    un = vecnorm(u, 2, 1);
    u  = u ./ max(un, 1e-300);  u(:, un < 1e-12) = 0;
    r1 = exp(-2 * pi * (pi - th) / V);
    r2 = exp(-2 * pi * (pi + th) / V);
    D  = th + (th - 2*pi) .* r1 + (th + 2*pi) .* r2;
    N  = (1 - th.^2 / V) + (1 - (th - 2*pi).^2 / V) .* r1 + (1 - (th + 2*pi).^2 / V) .* r2;
    dlogp = N ./ max(D, 1e-300) - cot(th);             % d/dtheta log p
    dlogp(th < 1e-6) = 0;
    b = -dlogp .* u;                                     % grad theta = -u
end

function D = ks2(a, b)
    % Two-sample KS statistic
    a = double(a(:));  b = double(b(:));
    [vs, idx] = sort([a; b]);
    lab = [ones(numel(a), 1) / numel(a); -ones(numel(b), 1) / numel(b)];
    cs  = cumsum(lab(idx));
    D   = max(abs(cs([diff(vs) ~= 0; true])));
end

function plot_comparison_results(T_cmp, cmpStore, cmpCfg)
    % Concentration, selection success, batch size, error distributions
    samplers = setdiff(unique(T_cmp.method, 'stable'), {'Multistart RGD', 'Target (exact, local)'}, 'stable');
    maxNs = max(T_cmp.Ns(~strcmp(T_cmp.method, 'Target (exact, local)')));
    pcol  = delta_name(cmpCfg.succ_deg);
    outVals = unique(T_cmp.outlierRatio)';
    cols = lines(numel(samplers) + 2);

    figure('Color', 'w', 'Position', [80 80 520 * numel(outVals) 420]);
    tiledlayout(1, numel(outVals), 'TileSpacing', 'compact');
    for o = outVals
        nexttile; hold on; box on; grid on;
        for m = 1:numel(samplers)
            idx = strcmp(T_cmp.method, samplers{m}) & T_cmp.Ns == maxNs & abs(T_cmp.outlierRatio - o) < 1e-12;
            G = groupsummary(T_cmp(idx, :), 'beta', 'mean', pcol);
            plot(G.beta, max(G.(['mean_' pcol]), 1e-6), '-o', 'Color', cols(m, :), 'LineWidth', 1.6, 'DisplayName', samplers{m});
        end
        idx = strcmp(T_cmp.method, 'Target (exact, local)') & abs(T_cmp.outlierRatio - o) < 1e-12;
        if any(idx)
            G = groupsummary(T_cmp(idx, :), 'beta', 'mean', pcol);
            plot(G.beta, G.(['mean_' pcol]), 'k--', 'LineWidth', 1.6, 'DisplayName', 'Exact target');
        end
        pu = T_cmp.pUnif(find(~isnan(T_cmp.pUnif), 1));
        plot(xlim, pu * [1 1], 'k:', 'LineWidth', 1.4, 'DisplayName', 'Uniform (Haar)');
        set(gca, 'YScale', 'log', 'XScale', 'log');
        xlabel('\beta');  ylabel(sprintf('Fraction of samples within %g deg of q^*', cmpCfg.succ_deg));
        title(sprintf('Concentration, outliers = %.2f, N_s = %d', o, maxNs));
        legend('Location', 'best');
    end

    betaVals = unique(T_cmp.beta);  beta0 = betaVals(ceil(numel(betaVals) / 2));
    ncols = arrayfun(@(n) sprintf('Psucc_n%d', n), cmpCfg.n_curve, 'UniformOutput', false);
    figure('Color', 'w', 'Position', [80 80 520 * numel(outVals) 420]);
    tiledlayout(1, numel(outVals), 'TileSpacing', 'compact');
    for o = outVals
        nexttile; hold on; box on; grid on;
        for m = 1:numel(samplers)
            idx = strcmp(T_cmp.method, samplers{m}) & T_cmp.Ns == maxNs & abs(T_cmp.beta - beta0) < 1e-12 & ...
                abs(T_cmp.outlierRatio - o) < 1e-12;
            if ~any(idx), continue; end
            y = mean(T_cmp{idx, ncols}, 1, 'omitnan');
            plot(cmpCfg.n_curve, y, '-o', 'Color', cols(m, :), 'LineWidth', 1.6, 'DisplayName', samplers{m});
        end
        idx = strcmp(T_cmp.method, 'Multistart RGD') & T_cmp.Ns == maxNs & abs(T_cmp.beta - beta0) < 1e-12 & ...
            abs(T_cmp.outlierRatio - o) < 1e-12;
        if any(idx)
            G = groupsummary(T_cmp(idx, :), 'K', 'mean', 'success');
            plot(G.K, G.mean_success, 'ks', 'MarkerFaceColor', 'k', 'DisplayName', 'Multistart RGD (x = K refinements)');
        end
        set(gca, 'XScale', 'log');  ylim([0 1.05]);
        xlabel('Batch size n (samplers)  /  K refined starts (RGD)');
        ylabel(sprintf('P(best-TLS pick within %g deg of truth)', cmpCfg.succ_deg));
        title(sprintf('Selection success, \\beta = %.2f, outliers = %.2f', beta0, o));
        legend('Location', 'best');
    end

    figure('Color', 'w', 'Position', [80 80 520 * numel(outVals) 420]);
    tiledlayout(1, numel(outVals), 'TileSpacing', 'compact');
    for o = outVals
        nexttile; hold on; box on; grid on;
        for m = 1:numel(samplers)
            idx = strcmp(T_cmp.method, samplers{m}) & T_cmp.Ns == maxNs & abs(T_cmp.outlierRatio - o) < 1e-12;
            G = groupsummary(T_cmp(idx, :), 'beta', 'median', 'n95');
            plot(G.beta, G.median_n95, '-o', 'Color', cols(m, :), 'LineWidth', 1.6, 'DisplayName', samplers{m});
        end
        pu = T_cmp.pUnif(find(~isnan(T_cmp.pUnif), 1));
        plot(xlim, batch_for_success(pu, 0.95) * [1 1], 'k:', 'LineWidth', 1.4, 'DisplayName', 'Uniform (Haar)');
        set(gca, 'YScale', 'log', 'XScale', 'log');
        xlabel('\beta');  ylabel(sprintf('Batch size for 95%% chance of a sample within %g deg', cmpCfg.succ_deg));
        title(sprintf('Batch size needed, outliers = %.2f (Inf not drawn)', o));
        legend('Location', 'best');
    end

    pick = find([cmpStore.beta] == beta0, 1);
    if ~isempty(pick)
        s = cmpStore(pick);
        figure('Color', 'w', 'Position', [80 80 560 420]); hold on; box on; grid on;
        for m = 1:numel(s.names)
            e = sort(double(s.errStar{m}));
            stairs(max(e, 1e-3), (1:numel(e)) / numel(e), 'LineWidth', 1.6, 'DisplayName', s.names{m});
        end
        if ~isempty(s.errTarget)
            e = sort(double(s.errTarget));
            stairs(max(e, 1e-3), (1:numel(e)) / numel(e), 'k--', 'LineWidth', 1.8, 'DisplayName', 'Exact target');
        end
        d = logspace(-3, log10(180), 200);
        plot(d, (deg2rad(d) - sin(deg2rad(d))) / pi, 'k:', 'LineWidth', 1.4, 'DisplayName', 'Uniform (Haar)');
        set(gca, 'XScale', 'log');  xlim([0.1 180]);
        xlabel('Rotation distance to q^* [deg]');  ylabel('ECDF');
        title(sprintf('Distribution around the minimiser, \\beta = %.2f, outliers = %.2f, repeat %d', s.beta, s.outlierRatio, s.repeat));
        legend('Location', 'southeast');
    end
end
