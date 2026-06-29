% Extended ASBS-M (Alg. 2)
% Stiefel manifold
% Verifying Eq. 37-38: convergence to Low energy

clear; clc; close all;
% Inverse temperature (higher = lower temp)
Beta = [0.001, 0.01, 0.01, 0.1, 0.5, 1.3, 2, 5, 7, 10, 20, 50, 100, 200, 1000, 10000, 1000000];
empirical_energy = zeros(length(Beta), 1);

% Repeat for different temperature regime
for beta_idx = 1:length(Beta)
    beta = Beta(beta_idx);
    fprintf('Running beta = %.5f\n', beta);

    % Controller, corrector w.r.t. current beta
    [netU, netH, params] = ASBS_Stiefel_sampler(beta);

    % Verification of Eq.38
    % Sample initial particles from uniform Haar (using QR)
    number_samples = 5000;
    X0 = zeros(params.n, params.p, number_samples);
    for i = 1:number_samples
        X0(:,:,i) = Retraction(randn(params.n, params.p));
    end 

    final_samples = simulate_trajectories(netU, X0, params.sigma, params.dt, ...
        params.N_steps, number_samples, params.n, params.p);
    X_final = final_samples(:,:,:, params.N_steps+1);
    
    % Empirical energy
    for i = 1:number_samples
        empirical_energy(beta_idx) = empirical_energy(beta_idx) + trace(X_final(:,:,i)' * params.H * X_final(:,:,i))';
    end
    empirical_energy(beta_idx) = empirical_energy(beta_idx) / number_samples;
    
    fprintf('Theoretical Minimum Energy (sum of lowest %d evals): %.2f\n', params.p, params.target_min_energy);
    fprintf('Trained Model Expected Energy E[tr(X^T H X)]: %.4f\n \n', empirical_energy(beta_idx));
end

%% Plot
% Use PDF export for the Figure and then Latex graphic to charge the image
fig = figure('Units', 'inches', 'Position', [2, 2, 6, 4.5], 'Color', 'w');

% Define colors
c_navy     = [0, 51, 102] / 255;
c_burgundy = [153, 51, 51] / 255;
c_grey     = [102, 102, 102] / 255;

hold on;

% Plotting
p1 = plot(Beta, empirical_energy, '-o', 'Color', c_navy, ...
    'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', [0.8, 0.9, 1.0]); 

p2 = plot(Beta, ones(size(Beta)) * params.target_min_energy, '--', ...
    'Color', c_burgundy, 'LineWidth', 2.5);

p3 = plot(Beta, ones(size(Beta)) * params.target_max_energy, '--', ...
    'Color', c_grey, 'LineWidth', 2.5);

% Axis Formatting
ax = gca;
ax.XScale = 'log';          % Sets log10 scale
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.LineWidth = 1.2;
ax.FontSize = 20;
ax.FontName = 'Times New Roman';
ax.Box = 'on';
grid on;
ax.GridAlpha = 0.2;

% Labels
xlabel('Scaled Inverse Temperature ($\beta$)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Expected Energy $\mathbb{E}[\mathrm{Tr}(X^{\top} H X)]$', 'Interpreter', 'latex', 'FontSize', 16);
title('Convergence to Low Energy Subspace', 'Interpreter', 'latex', 'FontSize', 16);

% Legend
leg = legend([p1, p2, p3], {'Empirical', 'Min', 'Max'}, ...
    'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
leg.Box = 'off';

%% Helper functions

function [netU, netH, params] = ASBS_Stiefel_sampler(beta)
    % Paramters
    params.n = 4;                   % Ambient dimension
    params.p = 2;                   % Subspace dimension
    params.B = 600;                 % Batch size (particles)
    params.N_steps = 199;           % SDE discretization step
    params.dt = 1/params.N_steps;   % time step
    params.sigma = 1.0;             % constant noise amplitude
    params.K_epochs = 1000;         % training epochs
    params.lr = 1e-3;               % learning rate
    
    % Hamiltonian and Energy landscape E(X) = beta * tr(X^T H X)
    params.H = [4,   0.5, 2.5, 1;
                0.5, 4,   1,   2.5;
                2.5, 1,   4,   0.5;
                1,   2.5, 0.5, 4];
    
    spec_H = eig(params.H);
    ordered_spec = sort(spec_H, 'ascend');
    % Minimum Energy (sum of p smallest)
    params.target_min_energy = sum(ordered_spec(1:params.p));
    % Maximum Energy p/n tr(H)
    params.target_max_energy = params.p / params.n * sum(spec_H);
    
    % Neural Networks architecture
    % controller u_theta = inputs [X_flat; t] -> Outputs X_flat
    layersU = [
        featureInputLayer(params.n*params.p + 1, 'Normalization', 'none')
        fullyConnectedLayer(256)
        tanhLayer
        fullyConnectedLayer(256)
        tanhLayer
        fullyConnectedLayer(params.n*params.p)
        ];
    netU = dlnetwork(layersU);
    
    % Corrector: h_phi: inputs [X_flat] -> Outputs X_flat
    layersH = [
        featureInputLayer(params.n*params.p, 'Normalization', 'none')
        fullyConnectedLayer(256)
        tanhLayer
        fullyConnectedLayer(256)
        tanhLayer
        fullyConnectedLayer(params.n*params.p)
        ];
    netH = dlnetwork(layersH);
    
    % Optimizer states for Adam
    [avgU, avgSqU, avgH, avgSqH] = deal([]);
    
    fprintf('Starting ASBS-M Training...\n');
    
    % Training Loop
    for k = 1:params.K_epochs
    
        % Sample initial particles from uniform Haar (using QR)
        X0 = zeros(params.n, params.p, params.B);
        for i = 1:params.B
            X0(:,:,i) = Retraction(randn(params.n, params.p));
        end
    
        % Simulate trajectories with current u_theta
        X_traj = simulate_trajectories(netU, X0, params.sigma, params.dt, ...
            params.N_steps, params.B, params.n, params.p);
    
        % Initialize terminal adjoitn
        X_N = X_traj(:,:,:,params.N_steps+1);
        h_val = predict_net(netH, X_N, params.n, params.p, params.B, false); % false = no time input
    
        % Euclidean gradient of E(x): 2 * beta * H * X_N
        gradE = 2 * beta * pagemtimes(params.H, X_N);
        v_curr = batchProjector(X_N, gradE) + h_val;
    
        % Backward sequential projection (PAT)
        V_traj = zeros(params.n, params.p, params.B, params.N_steps);
        for j = params.N_steps:-1:1
            X_j = X_traj(:,:,:,j);
            V_traj(:,:,:,j) = batchProjector(X_j, v_curr);
            v_curr = V_traj(:,:,:,j);
        end
    
        % Update controller u_theta
        % Flatten time and batch dimension for vector evaluation
        X_traj_flat = reshape(X_traj(:,:,:,1:params.N_steps), params.n, ...
            params.p, params.B * params.N_steps);
        V_traj_flat = reshape(V_traj, params.n, params.p, params.B * params.N_steps);
        T_traj = kron(0:params.dt:(1-params.dt), ones(1,params.B)); % Kronecker tensor product
    
        [gradU, lossU] = dlfeval(@computeLossU, netU, X_traj_flat, T_traj, ...
            V_traj_flat, params.sigma, params.n, params.p, params.B*params.N_steps);
        [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, k, params.lr);
    
        % Resample trajectories with updated theta
        X_resampled = simulate_trajectories(netU, X0, params.sigma, params.dt, ...
            params.N_steps, params.B, params.n, params.p);
        X_1 = X_resampled(:,:,:,params.N_steps + 1);
    
        % Compute projected chord
        chord = (X_1 - X0) / (params.sigma^2);
        b_target = - batchProjector(X_1, chord);
    
        % Update correct h_phi
        [gradH, lossH] = dlfeval(@computeLossH, netH, X_1, b_target, params.n, params.p, params.B);
        [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, k, params.lr);    
    
        if mod(k, 100) == 0 || k==1
            fprintf('Epoch %04d | Loss U: %.4f | Loss H: %.4f\n', k, extractdata(lossU), extractdata(lossH));
        end
    end
end

% Deep learning loss function executed via dlfeval
function [gradients, loss] = computeLossU(netU, X_batch, t_batch, V_target, sigma, n, p, BatchSize)
    % Format inputs for dlnetwork
    X_flat = reshape(X_batch, n*p, BatchSize);
    inputs = dlarray([X_flat; t_batch], 'CB');

    U_pred_flat = forward(netU, inputs);
    U_pred = reshape(U_pred_flat, n, p, BatchSize);
    
    X_dl = dlarray(X_batch);
    U_proj = batchProjector(X_dl, U_pred);
    V_dl = dlarray(V_target);

    % Loss || P_X(u) + sigma * v||^2
    residuals = U_proj + sigma * V_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
    gradients = dlgradient(loss, netU.Learnables);
end

function [gradients, loss] = computeLossH(netH, X_N, b_target, n, p, BatchSize)
    X_flat = reshape(X_N, n*p, BatchSize);
    inputs = dlarray(X_flat, 'CB');

    H_pred_flat = forward(netH, inputs);
    H_pred = reshape(H_pred_flat, n, p, BatchSize);

    X_dl = dlarray(X_N);
    H_proj = batchProjector(X_dl, H_pred);
    b_dl = dlarray(b_target);

    residuals = H_proj - b_dl;
    loss = sum(residuals.^2, 'all') / BatchSize;
    gradients = dlgradient(loss, netH.Learnables);
end

function X_traj = simulate_trajectories(net, X0, sigma, dt, N, B, n, p)
    X_traj = zeros(n, p, B, N+1);
    X_traj(:,:,:,1) = X0;

    for j = 1:N
        t = (j-1)*dt;
        X_curr = X_traj(:,:,:,j);

        % Predict control 
        u_val = predict_net(net, X_curr, n, p, B, true, t);
        noise = randn(n, p, B);

        drift = sigma * batchProjector(X_curr, u_val) * dt;
        diff_term= sigma * sqrt(dt) * batchProjector(X_curr, noise);
        step_update = X_curr + drift + diff_term;

        % Rectraction
        X_next = zeros(n, p, B);
        for i = 1:B
            X_next(:,:,i) = Retraction(step_update(:,:,i));
        end
        X_traj(:,:,:,j+1) = X_next;
    end
end

function out = predict_net(net, X, n, p, B, has_time, t)
    X_flat = reshape(X, n*p, B);
    if has_time
        inputs = dlarray([X_flat; repmat(t, 1, B)], 'CB');
    else
        inputs = dlarray(X_flat, 'CB');
    end
    out_flat = extractdata(predict(net, inputs));
    out = reshape(out_flat, n, p, B);
end

function P_Z = batchProjector(X, Z)
    % vectorized orthogonal projection
    % pagemtimes enables batch matrix multiplication ion Matlab
    X_T_Z = pagemtimes(X, 'transpose', Z, 'none');
    Z_T_X = pagemtimes(Z, 'transpose', X, 'none');
    sym_part = 0.5 * (X_T_Z + Z_T_X);
    P_Z = Z - pagemtimes(X, sym_part);
end

function Q = Retraction(A)
    % Retraction map using QR deconmposition
    [Q_full, R] = qr(A, 0);
    D = diag(sign(diag(R))); % enforce uniqueness of QR
    Q = Q_full * D;
end