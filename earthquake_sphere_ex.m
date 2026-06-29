% Earthquake Distribution with ASBS-M
clear; close all; clc;

% 1. Import and Prepare Data
fprintf('Start Import data...\n');
data = readtable('query.csv'); 

% Spherical to Cartesian Conversion
lat = deg2rad(data.latitude);
lon = deg2rad(data.longitude);

% X, Y, Z coordinates on the unit sphere
x_dir = cos(lat) .* cos(lon);
y_dir = cos(lat) .* sin(lon);
z_dir = sin(lat);
M_impact = [x_dir'; y_dir'; z_dir'];
valid_count = size(M_impact, 2);

% Train/Test Dataset Split
split_ratio = 0.7;
num_train = floor(valid_count * split_ratio);
M_train = M_impact(:, 1:num_train);
fprintf('Import data completed. Valid points: %d\n', valid_count);

%% 2. Run Sampler Training
[uNet, hNet, params] = ASBS_sphere_sampler(M_train);
fprintf('\nTraining completed.\n'); 

%% 3. Generate Final Samples
fprintf('Generating final manifold samples...\n');
N_samples = 700;
[~, M_generated] = simulate_trajectories_fixed(uNet, params, N_samples);

%% 4. Plot and Visualization
% Prepare Test Data
max_test_particles = N_samples;
test_end_idx = min(valid_count, num_train + max_test_particles);
M_test = M_impact(:, num_train+1:test_end_idx);

% Prepare Train Data (Max 1000 points)
max_train_plot = 1000;
num_train_plot = min(max_train_plot, size(M_train, 2));

% Uniformly subsample indices to keep a representative global distribution
idx_train_plot = round(linspace(1, size(M_train, 2), num_train_plot));
M_train_plot = M_train(:, idx_train_plot);

[sx, sy, sz] = sphere(500);

% FIGURE 1: Earth with Train Data (Journal Quality)
fig1 = figure('Color', 'w', 'Position', [100 100 800 700],'Renderer', 'opengl');

% Sphere
surf(sx, sy, sz, 'EdgeColor', 'none', 'FaceColor', [0.94 0.94 0.94], ...  
    'FaceAlpha', 0.6, 'FaceLighting', 'gouraud', 'AmbientStrength', 0.5, ...
    'DiffuseStrength', 0.7, 'SpecularStrength', 0.05);          
hold on;

% Add Coastlines (Earth)
load coastlines; 
r = 1.0; 
x_land = r * cosd(coastlat) .* cosd(coastlon);
y_land = r * cosd(coastlat) .* sind(coastlon);
z_land = r * sind(coastlat);

plot3(x_land, y_land, z_land, 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);

% Equator line for boundary reference
theta = linspace(0, 2*pi, 150);
plot3(cos(theta), sin(theta), zeros(size(theta)), ...
    '--', 'LineWidth', 1.2, 'Color', [0.55 0.55 0.55]);

% Plot: Train Data
hTrain = scatter3(M_train_plot(1,:), M_train_plot(2,:), M_train_plot(3,:), ...
    14, 'k', 'o', 'filled', 'MarkerEdgeColor', [1 1 1], 'LineWidth', 0.4, ...
    'MarkerFaceAlpha', 0.9, 'DisplayName', 'Train Data (Sampled)');

axis equal; axis tight;
xlim([-1.1 1.1]); ylim([-1.1 1.1]); zlim([-1.1 1.1]);

% Labels and Axis Properties
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 24);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 24);
zlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 24);
set(gca, 'FontSize', 16, 'LineWidth', 1.0, 'FontName', 'Times New Roman', ...
    'TickLabelInterpreter', 'latex', 'Box', 'off', 'GridAlpha', 0.25, ...
    'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
grid on;

% Lighting & Camera
camlight('headlight'); 
camlight('left'); 
lighting gouraud;
view(35, 30); 

% Legend & Title
legend(hTrain, 'Location', 'northeast', 'Interpreter', 'latex', ...
       'FontSize', 20, 'Box', 'off');
title('Earthquake Training Data Distribution', 'Interpreter', 'latex', 'FontSize', 24);

% FIGURE 2: Energy Landscape Heatmap with Test Data & Samples
% Evaluate the Energy Landscape Heatmap
grid_pts = [sx(:)'; sy(:)'; sz(:)'];

% Evaluate the energy 
E_vec = calc_E_vectorized(grid_pts, M_train, 600); 
E_matrix = reshape(E_vec, size(sx));

fig2 = figure('Color', 'w', 'Position', [150 150 800 700], 'Renderer', 'opengl');

% Sphere colored by the Energy Heatmap
surf(sx, sy, sz, E_matrix, 'EdgeColor', 'none', 'FaceAlpha', 0.75, ...              
    'FaceLighting', 'gouraud', 'AmbientStrength', 0.6, ...
    'DiffuseStrength', 0.7, 'SpecularStrength', 0.05);          
hold on;

% Apply colormap
colormap('cool');
cb = colorbar;
cb.TickLabelInterpreter = 'latex';
cb.Label.String = 'Energy $E(x)$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 24;
cb.Color = [0.15 0.15 0.15];
cb.Box = 'off';

% Equator line
theta = linspace(0, 2*pi, 150);
plot3(cos(theta), sin(theta), zeros(size(theta)), ...
    '--', 'LineWidth', 1.2, 'Color', [0.55 0.55 0.55]);

% Plot 1: True Data (Test Set in Black)
hTest = scatter3(M_test(1,:), M_test(2,:), M_test(3,:), 20, ...                             
    'k', 'o', 'filled', 'LineWidth', 0.4, 'MarkerFaceAlpha', 0.9, ...
    'DisplayName', 'True Data (Test Set)');

% Plot 2: ASBS-M Generated Samples (orange Triangles)
hSamples = scatter3(M_generated(1,:), M_generated(2,:), M_generated(3,:), ...
    24, [0.85, 0.35, 0.0], '^', 'filled', 'MarkerEdgeColor', [1, 1, 1], ...     
    'LineWidth', 0.4, 'MarkerFaceAlpha', 0.9, 'DisplayName', 'ASBS-M Samples');

axis equal; axis tight;
xlim([-1.1 1.1]); ylim([-1.1 1.1]); zlim([-1.1 1.1]);

% Labels and Axis Properties
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 24);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 24);
zlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 24);
set(gca, 'FontSize', 16, 'LineWidth', 1.0, 'FontName', 'Times New Roman', ...
    'TickLabelInterpreter', 'latex', 'Box', 'off', 'GridAlpha', 0.25, ...
    'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
grid on;

% Lighting & Camera
camlight('headlight'); 
camlight('left'); 
lighting gouraud;
view(35, 30); 

% Legend & Title
legend([hTest, hSamples], 'Location', 'northeast', 'Interpreter', 'latex', ...
       'FontSize', 22, 'Box', 'off');
title('Energy Landscape and Generated Samples', 'Interpreter', 'latex', 'FontSize', 24);

%%  HELPER FUNCTIONS

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
    
    % Hyperparameters
    params.B = 500;         
    params.epochs = 2000;  
    params.N_steps = 250;  
    params.dt = 1.0 / params.N_steps;
    params.max_drift = 30.0; % Strict gradient clipping bound
    
    % Sigma Schedule Formulation
    params.sig_a = 0.03;
    params.sig_b = 0.32;
    params.sigma_fn = @(t) params.sig_a + params.sig_b * (1 - t).^2;
    
    % Exact Definite Integral of Variance V(t) = int_0^t sigma_s^2 ds
    params.V_actual = @(t) (params.sig_a^2).*t ...
        + (2*params.sig_a*params.sig_b/3).*(1 - (1-t).^3) ...
        + (params.sig_b^2/5).*(1 - (1-t).^5);
    params.V1 = params.V_actual(1.0);
    
    % Kappa configuration
    params.kappa_end = 600;
    
    % RFF Setup
    num_freq = 256; 
    rff_bandwidth = 3.0; 
    params.B_u = randn(num_freq, 4) * rff_bandwidth; 
    params.B_u(:, 4) = randn(num_freq, 1) * 1.0; 
    params.B_h = randn(num_freq, 3) * rff_bandwidth;
    
    % Learning Rates
    lr_u_init = 1e-3;
    lr_h_init = 2e-3;
    decay_rate = 1;
    
    % Network Architecture
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
        lr_u = lr_u_init * (decay_rate ^ epoch); 
        lr_h = lr_h_init * (decay_rate ^ epoch);
        
        % STAIRCASE KAPPA ANNEALING
        if epoch <= params.epochs * 0.25
            curr_kappa = 150;
        elseif epoch <= params.epochs * 0.50
            curr_kappa = 300;
        elseif epoch <= params.epochs * 0.75
            curr_kappa = 450;
        else
            curr_kappa = params.kappa_end;
        end
        
        % Generate paths using fixed dt with drift clipping
        [X0, X1] = simulate_trajectories_fixed(uNet, params, params.B);
        
        % ADJOINT MATCHING
        [Xt, t_vec, a_t, sigma_t_vec] = compute_adjoint_targets(X0, X1, hNet, params, M_train, curr_kappa);
        
        dl_Xt_input_raw = [Xt; t_vec];
        dl_Xt_input_rff = dlarray(apply_rff(dl_Xt_input_raw, params.B_u), 'CB');
        dl_Xt_pos = dlarray(Xt, 'CB');
        dl_target_a = dlarray(a_t, 'CB');
        dl_sigma = dlarray(sigma_t_vec, 'CB');
        
        u_iter = u_iter + 1;
        [loss_u, grads_u] = dlfeval(@modelLossU, uNet, dl_Xt_input_rff, dl_target_a, dl_Xt_pos, dl_sigma);
        [uNet.Learnables, u_avgG, u_avgSqG] = adamupdate(uNet.Learnables, grads_u, u_avgG, u_avgSqG, u_iter, lr_u);
        
        % CORRECTOR MATCHING
        [X0_new, X1_new] = simulate_trajectories_fixed(uNet, params, params.B);
        b_target = compute_corrector_targets(X0_new, X1_new, params);
        
        dl_X1_input_raw = X1_new;
        dl_X1_input_rff = dlarray(apply_rff(dl_X1_input_raw, params.B_h), 'CB');
        dl_target_b = dlarray(b_target, 'CB');
        dl_X1_pos = dlarray(X1_new, 'CB'); 
        
        h_iter = h_iter + 1;
        [loss_h, grads_h] = dlfeval(@modelLossH, hNet, dl_X1_input_rff, dl_target_b, dl_X1_pos);
        [hNet.Learnables, h_avgG, h_avgSqG] = adamupdate(hNet.Learnables, grads_h, h_avgG, h_avgSqG, h_iter, lr_h);
        
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

function [Xt, t_vec, a_t, sigma_t_vec] = compute_adjoint_targets(X0, X1, hNet, params, M_train, kappa)
    t_vec = rand(1, params.B);
    sigma_t_vec = params.sigma_fn(t_vec);
    
    rff_input_X1 = apply_rff(X1, params.B_h);
    dl_X1 = dlarray(rff_input_X1, 'CB');
    h_val = extractdata(predict(hNet, dl_X1));
    h_val = h_val - sum(h_val .* X1, 1) .* X1; 
    
    grad_riem_E = calc_gradE_M_vectorized(X1, M_train, kappa);
    a_1 = grad_riem_E + h_val;
    
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
        
        % MAX(0, ...) to prevent complex numbers from floating point underflow
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
        d01_v = dot_01(valid_mask);
        
        u_rev_unnorm = x0_v - x1_v .* d01_v;
        u_rev = u_rev_unnorm ./ (vecnorm(u_rev_unnorm, 2, 1) + 1e-8);
        
        chord_term = th_v ./ params.V1; % Geodesic term
        
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