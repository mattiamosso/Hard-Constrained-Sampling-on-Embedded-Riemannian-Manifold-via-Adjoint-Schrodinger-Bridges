% Extended ASBS-M (Alg. 2)
% High-Dimensional Redundant Inverse Kinematics for Closed-Loop Chains
% 10-DOF planar robot with strict loop closure and obstacle avoidance.

clear; clc; close all;

%% 1. Configuration & Parameters
% Temperature regime
beta = 10.0;
fprintf('Starting ASBS-M Training for 10-DOF Robot at beta = %.1f\n', beta);

% Robot Parameters
params.d = 10;                        % Number of joints
params.L = ones(params.d, 1);         % Link lengths
params.xT = 8.0;                      % Target End-Effector X
params.yT = 0.0;                      % Target End-Effector Y
params.thetaT = 0.0;                  % Target End-Effector Orientation

% Obstacle & Energy Parameters
params.obs = [4, 1.5;                 % Obstacle 1 (x,y)
              6, -1.0];               % Obstacle 2 (x,y)
params.alpha = 20.0;                  % Repulsion barrier scale
params.sigma_obs = 1.0;               % Repulsion width
params.lambda = 0.05;                 % Weak prior weight
params.q_rest = zeros(params.d, 1);   % Resting posture

% ASBS-M SDE & Neural Net Parameters
params.B = 600;                       % Batch size (particles)
params.N_steps = 200;                 % SDE discretization steps
params.dt = 1/params.N_steps;         % Time step
params.sigma = 0.5;                   % Constant noise amplitude
params.K_epochs = 600;                % Training epochs
params.lr = 1e-3;                     % Learning rate

%% 2. Initialize Neural Networks
% Controller: u_theta (Inputs: [q; t] -> Output: q_dim)
layersU = [
    featureInputLayer(params.d + 1, 'Normalization', 'none')
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(params.d)
];
netU = dlnetwork(layersU);

% Corrector: h_phi (Inputs: [q] -> Output: q_dim)
layersH = [
    featureInputLayer(params.d, 'Normalization', 'none')
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(params.d)
];
netH = dlnetwork(layersH);

% Optimizer states
[avgU, avgSqU, avgH, avgSqH] = deal([]);

%% 3. ASBS-M Training Loop (Algorithm 2)
for k = 1:params.K_epochs
    % 1. Sample initial particles (q0) and enforce constraint via Retraction
    % We use a slight random spread around an arching shape to aid initial roots
    guess = randn(params.d, params.B) * 0.2 + 0.1;
    q0 = Retraction(guess, params);
    
    % 2. Simulate forward trajectories with current u_theta
    q_traj = simulate_trajectories(netU, q0, params);
    
    % 3. Initialize terminal adjoint: v_N = P_{X_N} gradE(X_N) + h_phi(X_N)
    q_N = q_traj(:,:, params.N_steps + 1);
    h_val = predict_net(netH, q_N, params.d, params.B, false, 0);
    
    % Compute Euclidean gradient of E(q) via Auto-Diff
    [gradE, ~] = compute_energy_grad(q_N, beta, params);
    v_curr = batchProjector(q_N, gradE, params) + h_val;
    
    % 4. Backward Sequential Projection (PAT)
    V_traj = zeros(params.d, params.B, params.N_steps);
    for j = params.N_steps:-1:1
        q_j = q_traj(:,:, j);
        V_traj(:,:, j) = batchProjector(q_j, v_curr, params);
        v_curr = V_traj(:,:, j);
    end
    
    % 5. Update Controller (u_theta)
    q_traj_flat = reshape(q_traj(:,:, 1:params.N_steps), params.d, params.B * params.N_steps);
    V_traj_flat = reshape(V_traj, params.d, params.B * params.N_steps);
    T_traj = kron(0:params.dt:(1-params.dt), ones(1, params.B));
    
    [gradU, lossU] = dlfeval(@computeLossU, netU, q_traj_flat, T_traj, ...
        V_traj_flat, params.sigma, params.B * params.N_steps, params);
    [netU, avgU, avgSqU] = adamupdate(netU, gradU, avgU, avgSqU, k, params.lr);
    
    % 6. Resample trajectories with updated controller
    q_resampled = simulate_trajectories(netU, q0, params);
    q_1 = q_resampled(:,:, params.N_steps + 1);
    
    % 7. Compute Projected Chord and Update Corrector (h_phi)
    chord = (q_1 - q0) / (params.sigma^2);
    b_target = -batchProjector(q_1, chord, params);
    
    [gradH, lossH] = dlfeval(@computeLossH, netH, q_1, b_target, params.d, params.B, params);
    [netH, avgH, avgSqH] = adamupdate(netH, gradH, avgH, avgSqH, k, params.lr);    
    
    if mod(k, 50) == 0 || k == 1
        fprintf('Epoch %04d | Loss U: %.4f | Loss H: %.4f\n', k, extractdata(lossU), extractdata(lossH));
    end
end

%% 4. Sampling & Visualization
fprintf('\nGenerating Multi-Modal Valid Configurations...\n');
N_test = 100;
guess_test = randn(params.d, N_test) * 0.3 + 0.1;
q0_test = Retraction(guess_test, params);
final_trajs = simulate_trajectories(netU, q0_test, params);
q_final = final_trajs(:,:, end);

% Set up figure
fig = figure('Units', 'inches', 'Position', [1, 1, 8, 6], 'Color', 'w');
hold on; axis equal; box on;
title('\textbf{Multi-Modal Valid Configurations of a Closed-Loop Chain}', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$X$ Workspace', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$Y$ Workspace', 'Interpreter', 'latex', 'FontSize', 14);

% Plot Obstacles
theta_circ = linspace(0, 2*pi, 100);
for k = 1:size(params.obs, 1)
    Ox = params.obs(k, 1); Oy = params.obs(k, 2);
    fill(Ox + params.sigma_obs*cos(theta_circ), Oy + params.sigma_obs*sin(theta_circ), ...
        [0.7 0.7 0.7], 'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 1.5, 'FaceAlpha', 0.5);
    text(Ox, Oy, sprintf('$\\mathcal{O}_%d$', k), 'Interpreter', 'latex', ...
        'FontSize', 14, 'HorizontalAlignment', 'center');
end

% Plot Target End-Effector Block
plot(params.xT, params.yT, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', '#FFD700', 'LineWidth', 1.5);
text(params.xT + 0.3, params.yT, '$c(q)=0$', 'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');

% Draw Sampled Robots
c_blue = [0, 0.4470, 0.7410];
c_red  = [0.8500, 0.3250, 0.0980];
for i = 1:N_test
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q_final(:, i), params);
    x_coords = [0; x_j]; y_coords = [0; y_j];
    
    % Color by mode (did the middle joint go above or below the obstacles?)
    if y_j(5) > 1.0
        c_line = c_blue; 
    else
        c_line = c_red;
    end
    
    % Draw links
    plot(x_coords, y_coords, '-', 'Color', [c_line, 0.15], 'LineWidth', 2);
    % Draw joints
    plot(x_coords, y_coords, 'o', 'Color', [c_line, 0.2], 'MarkerSize', 3, 'MarkerFaceColor', 'w');
end

% Highlight one specific modes for clarity
highlight_idx = 2; % Arbitrary indices
for i = 1:length(highlight_idx)
    idx = highlight_idx(i);
    [~, ~, x_j, y_j] = robot_kinematics_and_jacobian(q_final(:, idx), params);
    x_coords = [0; x_j]; y_coords = [0; y_j];
    
    if y_j(5) > 1.0, c_line = c_blue; else, c_line = c_red; end
    
    plot(x_coords, y_coords, '-', 'Color', c_line, 'LineWidth', 3);
    plot(x_coords, y_coords, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
end

% Plot Base
plot(0, 0, 'k^', 'MarkerSize', 14, 'MarkerFaceColor', '#D3D3D3', 'LineWidth', 1.5);
xlim([-1, 10]); ylim([-3, 5]);

%% Helper Functions

function [c, J, x_j, y_j] = robot_kinematics_and_jacobian(q, params)
    % Evaluates Forward Kinematics, Constraints, and Jacobian
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
    
    % Constraints (m=3) with atan2 trick for 2pi invariance
    x_tip = x_j(d, :); y_tip = y_j(d, :); theta_tip = theta(d, :);
    c1 = x_tip - params.xT;
    c2 = y_tip - params.yT;
    c3 = atan2(sin(theta_tip - params.thetaT), cos(theta_tip - params.thetaT));
    c = [c1; c2; c3];
    
    % Jacobian Matrix J (3 x d)
    J = zeros(3, d, B);
    for k = 1:d
        if k == 1
            x_prev = zeros(1, B); y_prev = zeros(1, B);
        else
            x_prev = x_j(k-1, :); y_prev = y_j(k-1, :);
        end
        % J = [ - (y_tip - y_prev);  (x_tip - x_prev);  1 ]
        J(1, k, :) = -(y_tip - y_prev);
        J(2, k, :) =  (x_tip - x_prev);
        J(3, k, :) =  1; % Derivative of atan2(sin(e), cos(e)) is exactly 1
    end
end

function q_new = Retraction(q, params)
    % Newton-Raphson Projection onto c(q) = 0
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
        JJT = pagemtimes(J, J_T) + 1e-5 * repmat(eye(3), 1, 1, B); % Damping
        step = pagemtimes(J_T, pagemtimes(pageinv(JJT), c_mat));
        q_curr = q_curr - reshape(step, params.d, B);
    end
    q_new = q_curr;
end

function Pv = batchProjector(q, v, params)
    % Orthogonal projection of v onto the tangent space T_q M
    [~, J_num] = robot_kinematics_and_jacobian(q, params);
    B = size(q, 2);
    
    % Compute P = I - J^T (JJ^T)^-1 J numerically to avoid dlarray issues
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
    % Wrapper to execute dlgradient
    q_dl = dlarray(q_batch, 'CB');
    [E_val, gradE] = dlfeval(@energy_and_grad, q_dl, beta, params);
    gradE = extractdata(gradE);
    E_val = extractdata(E_val);
end

function [E_val, gradE] = energy_and_grad(q_dl, beta, params)
    B = size(q_dl, 2);
    
    % Replace cumsum with a loop to support dlarray in all MATLAB versions
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
    % Obstacle Repulsion
    for k = 1:size(params.obs, 1)
        dist_sq = (x_j - params.obs(k,1)).^2 + (y_j - params.obs(k,2)).^2;
        E = E + sum(params.alpha * exp(-dist_sq / (2 * params.sigma_obs^2)), 1);
    end
    % Weak Prior
    E = E + params.lambda * sum((q_dl - params.q_rest).^2, 1);
    
    E_beta = beta * E;
    gradE = dlgradient(sum(E_beta), q_dl);
    E_val = E_beta;
end

function [gradients, loss] = computeLossU(netU, X_batch, t_batch, V_target, sigma, BatchSize, params)
    inputs = dlarray([X_batch; t_batch], 'CB');
    U_pred = forward(netU, inputs);
    U_proj = batchProjector(X_batch, U_pred, params); % Safe projection
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

function q_traj = simulate_trajectories(net, q0, params)
    B = size(q0, 2);
    q_traj = zeros(params.d, B, params.N_steps + 1);
    q_traj(:,:,1) = q0;
    
    for j = 1:params.N_steps
        t = (j-1) * params.dt;
        q_curr = q_traj(:,:,j);
        
        u_val = predict_net(net, q_curr, params.d, B, true, t);
        noise = randn(params.d, B);
        
        drift = params.sigma * batchProjector(q_curr, u_val, params) * params.dt;
        diff_term = params.sigma * sqrt(params.dt) * batchProjector(q_curr, noise, params);
        
        step_update = q_curr + drift + diff_term;
        q_traj(:,:,j+1) = Retraction(step_update, params); % Enforce hard constraint
    end
end

function out = predict_net(net, X, d, B, has_time, t)
    if has_time
        inputs = dlarray([X; repmat(t, 1, B)], 'CB');
    else
        inputs = dlarray(X, 'CB');
    end
    out = extractdata(predict(net, inputs));
end