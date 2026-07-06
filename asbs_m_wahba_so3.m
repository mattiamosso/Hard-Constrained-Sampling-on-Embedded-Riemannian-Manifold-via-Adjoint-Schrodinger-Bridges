% Extended-ASBS-M (Alg.2) for robust Wahba on SO(3)

% Notes:
%   The network is retrained for each beta
%   Ns is evaluated cheaply by generating max(NsList) samples once and then
%   taking prefixes of the same sample pool

%% 1. Base problem and algorithm configuration
baseParams = default_wahba_params();

%% 2. Experiment grid
expCfg.betas = [0.25 0.50 1.00 2.00 4.00];
expCfg.NsList = [500 1000 2500 5000 10000];
expCfg.outlierRatios = [0.25 0.50 0.75 0.85];
expCfg.nRepeats = 5;              
expCfg.seed0 = 1100;
expCfg.saveCsv = true;
expCfg.makePlots = false;

%% 3. Run experiments
T = run_asbs_experiment_grid(baseParams, expCfg);

disp(T);

if expCfg.saveCsv
    writetable(T, 'asbs_wahba_experiment_results.csv');
    fprintf('\nSaved results to asbs_wahba_experiment_results.csv\n');
end

if expCfg.makePlots
    plot_asbs_experiment_results(T);
end

%% Experiment functions

function params = default_wahba_params()
    params.useSynthetic = true;

    % Synthetic robust Wahba setup.
    params.Ncorr = 1000;
    params.outlierRatio = 0.75;
    params.inlierNoise = 0.10;
    params.sigmaMeas = 0.10;
    params.trueAxis = [0.35; -0.75; 0.56];
    params.trueAngleDeg = 72;

    % Robust TLS objective parameters.
    params.cbar = 2.7955;
    params.beta = 1.0;
    params.useSmoothTLS = true;
    params.tlsSmoothing = 0.20;

    % ASBS-M parameters.
    params.d = 4;
    params.B = 512;
    params.N_steps = 150;
    params.dt = 1 / params.N_steps;
    params.sigma = 0.50;
    params.K_epochs = 600;
    params.lr = 2e-4;
    params.gradClip = 50;
    params.hiddenWidth = 256;
    params.printEvery = 50;
    params.verbose = true;

    % Sampling and diagnostics.
    params.N_test = 10000;
    params.N_random_baseline = 10000;
end

function T = run_asbs_experiment_grid(baseParams, expCfg)
    T = table();
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

                % One seed per setting. This controls the synthetic instance,
                % initialization, SDE noise, and sample generation.
                runSeed = expCfg.seed0 + 100000 * rep + 1000 * round(100 * outlierRatio) + round(100 * beta);
                rng(runSeed);

                fprintf('\n============================================================\n');
                fprintf('Run %d / %d | repeat %d | beta %.3f | outliers %.2f\n', ...
                    runId, totalRuns, rep, beta, outlierRatio);
                fprintf('============================================================\n');

                T_run = run_single_asbs_setting(params, expCfg.NsList, rep, runSeed);
                T = [T; T_run]; %#ok<AGROW>
            end
        end
    end
end

function T = run_single_asbs_setting(params, NsList, rep, runSeed)
    if params.useSynthetic
        params = makeSyntheticWahba(params);
    else
        % To handle real dataset
        params.A = normalizeColumns(params.A);  % 3xN matrix of source unit vectors ai
        params.Bmeas = normalizeColumns(params.Bmeas);  % 3xN measured unit vectors bi
        params.sigmas = params.sigmas(:)';  % contains noise scales sigmai
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

    [netU, trainInfo] = train_asbs_m(params);

    fprintf('\nSampling terminal rotations without local refinement...\n');
    maxNs = max(NsList);

    q0_test = uniformQuaternions(maxNs);
    final_trajs = simulate_trajectories(netU, q0_test, params);
    q_samples = final_trajs(:, :, end);
    F_samples = wahba_tls_energy(q_samples, params, false);

    q_random = uniformQuaternions(maxNs);
    F_random = wahba_tls_energy(q_random, params, false);

    T = summarize_sample_prefixes(params, NsList, q_samples, F_samples, q_random, F_random, ...
        trueTLS, rep, runSeed, trainInfo);
end

function [netU, trainInfo] = train_asbs_m(params)
    [netU, netH] = initialize_networks(params);

    avgU = []; avgSqU = [];
    avgH = []; avgSqH = [];
    lossUHist = zeros(params.K_epochs, 1);
    lossHHist = zeros(params.K_epochs, 1);
    bestTrainHist = zeros(params.K_epochs, 1);

    for k = 1:params.K_epochs
        q0 = uniformQuaternions(params.B);

        q_traj = simulate_trajectories(netU, q0, params);

        q_N = q_traj(:, :, params.N_steps + 1);
        h_val = predict_net(netH, q_N, params.d, params.B, false, 0);
        [gradE, ~] = compute_energy_grad(q_N, params);
        v_curr = batchProjector(q_N, gradE + h_val, params);

        V_traj = zeros(params.d, params.B, params.N_steps);
        for j = params.N_steps:-1:1
            q_j = q_traj(:, :, j);
            V_traj(:, :, j) = batchProjector(q_j, v_curr, params);
            v_curr = V_traj(:, :, j);
        end

        q_traj_flat = reshape(q_traj(:, :, 1:params.N_steps), params.d, params.B * params.N_steps);
        V_traj_flat = reshape(V_traj, params.d, params.B * params.N_steps);
        T_traj = kron(0:params.dt:(1 - params.dt), ones(1, params.B));

        [gradU, lossU] = dlfeval(@computeLossU, netU, q_traj_flat, T_traj, ...
            V_traj_flat, params.sigma, params.B * params.N_steps, params);
        gradU = clipGradients(gradU, params.gradClip);
        [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, k, params.lr);

        q_resampled = simulate_trajectories(netU, q0, params);
        q_1 = q_resampled(:, :, params.N_steps + 1);
        chord = (q_1 - q0) / (params.sigma^2);
        b_target = -batchProjector(q_1, chord, params);

        [gradH, lossH] = dlfeval(@computeLossH, netH, q_1, b_target, params.d, params.B, params);
        gradH = clipGradients(gradH, params.gradClip);
        [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, k, params.lr);

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
    % The plots use per-setting means across repeats. If nRepeats = 1, this is just the raw value.
    G = groupsummary(T, {'beta', 'outlierRatio', 'Ns'}, 'mean', ...
        {'bestASBS_TLS', 'bestRandom_TLS', 'bestASBS_rotErrDeg', ...
         'medianSample_rotErrDeg', 'fracSamplesBelow5Deg'});

    % 1. beta effect at max Ns, one curve per outlier ratio.
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

    % 2. Ns effect at the middle beta, one curve per outlier ratio.
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

    % 3. Concentration diagnostic
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

    % 4. Outlier robustness at max Ns and middle beta.
    figure('Color', 'w'); hold on; box on; grid on;
    idx = G.Ns == maxNs & abs(G.beta - beta0) < 1e-12;
    plot(G.outlierRatio(idx), G.mean_bestASBS_rotErrDeg(idx), '-o');
    xlabel('Outlier ratio');
    ylabel('Best ASBS rotation error [deg]');
    title(sprintf('Outlier robustness at beta = %.2f, Ns = %d', beta0, maxNs));
end

function make_wahba_two_panel_figure(params, q_samples, q_refined)
    % The figure has two horizontal panels:
    %   (a) exact TLS energy on a 2D geodesic slice of S^3 around q_true or q_refined;
    %   (b) S^2 directional visualization of two inlier vector correspondences,
    %       their exact rotation arcs, and the true rotation axis.
    
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
    
    % Use the true quaternion as the landscape center when available.
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
    
    % Panel (a): 2D geodesic slice of S^3
    nexttile(1);
    ng = 151;
    rmax = 1.20; % radians on S^3, large enough to show local basin
    [xg, yg] = meshgrid(linspace(-rmax, rmax, ng), linspace(-rmax, rmax, ng));
    U = null(q_center'); % 4 x 3 orthonormal tangent basis at q_center
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
    
    % Project a limited number of generated samples onto the same tangent slice.
    if ~isempty(q_samples)
        ns = min(size(q_samples, 2), 700);
        idx = round(linspace(1, size(q_samples, 2), ns));
        coords = project_to_tangent_slice(q_samples(:, idx), q_center, u1, u2);
        keep = abs(coords(1,:)) <= rmax & abs(coords(2,:)) <= rmax;
        plot(coords(1, keep), coords(2, keep), '.', 'Color', [0 0 0], ...
            'MarkerSize', 4, 'DisplayName', 'ASBS-M samples');
    end
    
    % Panel (b): S^2 directional correspondences and exact rotation
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
    
    % Choose two true inliers.
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
        0.000 0.447 0.741;   % deep blue
        0.850 0.325 0.098    % vermilion
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
    
        % Starting direction a_i.
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
    
        % Exact rotated direction R_true a_i.
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
    
        % Measured inlier endpoint.
        plot3(ax, ...
            b_meas(1), b_meas(2), b_meas(3), ...
            '^', ...
            'Color', c, ...
            'MarkerFaceColor', c, ...
            'MarkerSize', 8.5, ...
            'LineWidth', 1.4);
    
        % Exact rotation trajectory t -> R(t) a_i.
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

        drift = params.sigma * batchProjector(q_curr, u_val, params) * params.dt;
        diffTerm = params.sigma * sqrt(params.dt) * batchProjector(q_curr, noise, params);
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
    % Rotate fixed 3-vector a by each scalar-first quaternion in q.
    % q: 4 x B, a: 3 x 1, y: 3 x B.
    a1 = a(1); a2 = a(2); a3 = a(3);
    qw = q(1, :);
    qx = q(2, :); qy = q(3, :); qz = q(4, :);

    % t = 2 * cross(q_vec, a)
    tx = 2 * (qy * a3 - qz * a2);
    ty = 2 * (qz * a1 - qx * a3);
    tz = 2 * (qx * a2 - qy * a1);

    % y = a + qw * t + cross(q_vec, t)
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
    residuals = U_proj + sigma * V_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
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
