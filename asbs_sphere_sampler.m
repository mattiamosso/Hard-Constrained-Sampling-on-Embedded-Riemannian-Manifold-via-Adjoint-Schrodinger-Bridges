% Alg. 1 (two poles sphere split, known E(x))
% Simple case: Bi-modal distribution
clear; close all; clc;

[uNet, params] = ASBS_sphere_sampler();

%% Samples N_samples particles
fprintf('\nTraining completed. Generating final manifold samples...\n');
N_samples = 1500;
[~, X_test] = simulate_trajectories(uNet, params, N_samples);
    
%% Plot and Visualization
figure('Color','w','Position',[100 100 1100 850],'Renderer','opengl');

% High-resolution sphere
[sx,sy,sz] = sphere(500);
E_surface = 6*(1-sz.^2);

% Custom Plasma Colormap
anchor = [...
    0.0504 0.0298 0.5280
    0.2546 0.0139 0.6154
    0.4176 0.0006 0.6584
    0.5627 0.0515 0.6415
    0.6928 0.1651 0.5645
    0.7982 0.2802 0.4695
    0.8814 0.3925 0.3832
    0.9492 0.5178 0.2957
    0.9883 0.6523 0.2114
    0.9946 0.8228 0.1439];

n = 256;
x  = linspace(0,1,size(anchor,1));
xi = linspace(0,1,n);

cmap = interp1(x,anchor,xi,'pchip');
colormap(cmap);

% Surface
hSurf = surf(sx,sy,sz,E_surface,...
    'EdgeColor','none',...
    'FaceAlpha',0.45,...
    'FaceLighting','gouraud',...
    'AmbientStrength',0.55,...
    'DiffuseStrength',0.85,...
    'SpecularStrength',0.12);

hold on

% Optional contour lines
contour3(sx,sy,sz,E_surface, 12, 'k', 'LineWidth',0.8);

% Samples
hSamples = scatter3(X_test(1,:), X_test(2,:), X_test(3,:),...
    14, 'k', 'filled', 'DisplayName','ASBS-M Samples');

% Colormap
colormap(cmap)
clim([0 6])
cb = colorbar;

cb.FontSize = 24;
cb.FontName = 'Times New Roman';
cb.Label.String = 'Energy $E(x)$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 24;

% Axes
axis equal
axis tight

xlabel('$x_1$', 'Interpreter','latex', 'FontSize',28);
ylabel('$x_2$', 'Interpreter','latex', 'FontSize',28);
zlabel('$x_3$', 'Interpreter','latex', 'FontSize',28);

set(gca, 'FontSize',24, 'LineWidth',1.5, 'FontName','Times New Roman',...
    'TickLabelInterpreter','latex','Box','on');

% Lighting
camlight('headlight')
camlight('right')
camlight('left')
lighting gouraud

% Camera
view(40,25)

% Legend
legend(hSamples, 'Location','northeast', 'Interpreter','latex', 'FontSize',24, 'Box','off');

% Grid
grid on
set(gca,'GridAlpha',0.15)

title('Bi-Modal Distribution', 'Interpreter', 'latex', 'FontSize', 24);

%% Functions

function [uNet, params] = ASBS_sphere_sampler()
    % Algorithm A (ASBS-M) on S^2 subset R^3
    % Required MATLAB Deep Learning Toolbox

    % Fix seed for reproducibility
    rng(1, 'twister');

    % Hyperparameters
    params.B = 500;         % Batch size (number of trajectories)
    params.epochs = 600;    % Number of optimization iterations
    params.N_steps = 500;   % SDE time-discretization steps
    params.dt = 1.0 / params.N_steps;
    params.sigma = 1.0;     % Constant noise schedule amplitude

    % Learning rates for ADAM updates
    lr_u = 0.002;
    lr_h = 0.002;

    % u_theta(x, t) Network architecture
    layers_u = [
        featureInputLayer(4, 'Name', 'input_ut') % Inputs: [x1, x2, x3, t]
        fullyConnectedLayer(64, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(3, 'Name', 'fc3') % Ambient 3D output [u1, u2, u3]
        ];
    uNet = dlnetwork(layers_u);
    uNet = initialize(uNet);

    % h_phi(x) Network achitecture
    layers_h = [
        featureInputLayer(3, 'Name', 'input_hx') % Inputs: [x1, x2, x3]
        fullyConnectedLayer(48, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(48, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(3, 'Name', 'fc3') % Ambient 3D output
        ];
    hNet = dlnetwork(layers_h);
    hNet = initialize(hNet);

    % Initialize ADAM tracking structure
    u_avgG = []; u_avgSqG = []; u_iter = 0;
    h_avgG = []; h_avgSqG = []; h_iter = 0;

    fprintf('Starting ASBS-M ...\n');

    % Main Training Loop
    for epoch = 1:params.epochs
        % STEP 1: Adjoint Update
        % 1. Sample terminal points by running the controlled SDE on M
        % Note one could create a large bactch of trajectories and then samples uniformly from it
        [X0, X1] = simulate_trajectories(uNet, params, params.B);
        
        % 2. Samples bridges intermediates (Xt) and compute regresion target (a_t)
        [Xt, t_vec, a_t] = compute_adjoint_targets(X0, X1, hNet, params);
        dl_Xt_input = dlarray([Xt; t_vec], 'CB');
        dl_Xt_pos = dlarray(Xt, 'CB');
        dl_target_a = dlarray(a_t, 'CB');

        % 3. Update controller network
        u_iter = u_iter + 1;
        [loss_u, grads_u] = dlfeval(@modelLossU, uNet, dl_Xt_input, ...
            dl_target_a, dl_Xt_pos, params.sigma);
        [uNet.Learnables, u_avgG, u_avgSqG] = adamupdate(uNet.Learnables, ...
            grads_u, u_avgG, u_avgSqG, u_iter, lr_u);

        % STEP 2: Corrector Update
        % 1. Re-sample / refresh pairs using current controller
        [X0_new, X1_new] = simulate_trajectories(uNet, params, params.B);
        
        % 2. Compute exact corrector target via Riemannian Log Map
        b_target = compute_corrector_targets(X0_new, X1_new, params);

        % Prepare inputs for corrector optimization
        dl_X1_input = dlarray(X1_new, 'CB');
        dl_target_b = dlarray(b_target, 'CB');

        % 3. Update corrector network
        h_iter = h_iter + 1;
        [loss_h, grads_h] = dlfeval(@modelLossH, hNet, dl_X1_input, dl_target_b);
        [hNet.Learnables, h_avgG, h_avgSqG] = adamupdate(hNet.Learnables, ...
            grads_h, h_avgG, h_avgSqG, h_iter, lr_h);

        % Display convergence properties
        if mod(epoch, 100) == 0 || epoch == 1
            fprintf('Epoch %4d/%4d | Controller Loss: %8.5f | Corrector Loss: %8.5f\n', ...
                epoch, params.epochs, extractdata(loss_u), extractdata(loss_h));
        end
    end
end

function [X0, X1] = simulate_trajectories(uNet, params, num_samples)
    % Geodesic Random Walk with retraction instead of Exp. function
    % Integrate the forward SDE with immediate projection
    
    % Initialization
    % We sample from a Gaussian => Initial Uniform distribution on S^2
    X0 = randn(3, num_samples);     % Gaussian N(0, I_3)
    X0 = X0 ./ vecnorm(X0, 2, 1);   % Valid only for a unit sphere (otherwise multiply by R) 
    X1 = X0;    % used in the following and updata as current state

    for step = 1:params.N_steps
        t_val = (step - 1) * params.dt;
        
        % Use the current control evaluated 
        dl_in = dlarray([X1; t_val * ones(1, num_samples)], 'CB'); % Channel-Batch
        u_val = extractdata(predict(uNet, dl_in));
        
        % The control is in R^d we force it to the tangent space Tx S^2
        % u_tang. = u - u_radial, where u_radial = (u * x1) * x1 /|x1|^2
        % since we operate in a unit sphere => |x1|^2 = 1
        u_val = u_val - sum(u_val .* X1, 1) .* X1;

        noise = randn(3, num_samples);
        tangent_noise = noise - sum(noise .* X1, 1) .* X1;

        % Integration step (Euler-Maruyama)
        X1 = X1 + u_val * params.dt + params.sigma * sqrt(params.dt) * tangent_noise;

        % Retraction (equiv. Newton-Raphson Projection to correct discrete drift)
        X1 = X1 ./ vecnorm(X1, 2, 1);
    end
end

function [Xt, t_vec, a_t] = compute_adjoint_targets(X0, X1, hNet, params)
    % Implements Geodesic bridge and Parallel transport
    
    % initialize
    Xt = zeros(3, params.B);
    t_vec = rand(1, params.B);
    a_t = zeros(3, params.B);

    % Evaluate current corrector network at terminal positions
    dl_X1 = dlarray(X1, 'CB');
    h_val = extractdata(predict(hNet, dl_X1));
    h_val = h_val - sum(h_val .* X1, 1) .* X1; % tangent projection

    for i=1:params.B
        x0 = X0(:, i); % Particle i in the batch B (intial state)
        x1 = X1(:, i);
        t = t_vec(i);  % Random time i between 0-1

        % Spherical arc distance between x0, x1 (valid since unit sphere)
        dot_01 = max(min(dot(x0, x1), 1.0), -1.0); % safe clip top value applied
        theta = acos(dot_01);

        % Double-Well target Energy potential
        % E(x) = 6 * (1 - x3^2)
        grad_ambient_E = [0; 0; -12 * x1(3)];
        grad_riem_E = grad_ambient_E - dot(grad_ambient_E, x1) * x1;

        % Terminal value adjoint 
        a_1 = grad_riem_E + h_val(:, i);
        
        % avoid division 0/0 fro close initial and final value
        if theta < 1e-5
            Xt(:, i) = x1;
            a_t(:, i) = a_1;
            continue;
        end

        % Construct the geodesic bridge
        % geodesic direction i.e.,
        % unitary tangent vector pointing towards the shortest pathway towards x1 
        u_dir = (x1 - dot_01 * x0) / norm(x1 - dot_01 * x0);

        % mu_t interpolates along this circle path at fractional time t
        mu_t = x0 * cos(t * theta) + u_dir * sin(t * theta);

        % Adding localized tangential guassian noise
        noise = randn(3, 1);
        tangent_noise = noise - dot(noise, mu_t) * mu_t;
        bridge_std = params.sigma * sqrt(t * (1.0 - t)); % since T=1

        % Retraction onto the sphere surface
        xt_unnorm = mu_t + bridge_std * tangent_noise;
        xt = xt_unnorm / norm(xt_unnorm);
        Xt(:, i) = xt;
    
        % Geodesic Parallel Transport (X1 -> Xt)
        dot_1t = max(min(dot(x1, xt), 1.0), -1.0);
        phi = acos(dot_1t);

        if phi > 1e-05
            u_pt = (xt - dot_1t * x1) / norm(xt - dot_1t * x1);

            % Decompose vector along tracking arcs
            a1_parallel = dot(a_1, u_pt) * u_pt;
            a1_ortho = a_1 - a1_parallel;

            % Execute continuous coordinate rotation map matrix transformation
            a_t(:, i) = a1_ortho + a1_parallel * cos(phi) - dot(a_1, u_pt) * x1 *sin (phi);
        else
            a_t(:, i) = a_1;
        end
    end
end

function b_target = compute_corrector_targets(X0, X1, params)
    % Exact inverse exponential (logarithmic) Mapping
    b_target = zeros(3, params.B);
    integral_sigma2 = params.sigma^2; % int_0^1 sigma_t^2 dt (noise constant here)

    for i = 1:params.B
        x0 = X0(:, i);
        x1 = X1(:, i);

        dot_01 = max(min(dot(x1, x0), 1.0), -1.0);
        theta = acos(dot_01);

        if theta > 1e-05
            % Direction vector in T_X1 S^2 pointing to X0
            u_rev = (x0 - dot_01 * x1) / norm(x0 - dot_01 * x1);
            log_map_x1_x0 = u_rev  * theta;

            % Target value assignment
            b_target(:, i) = log_map_x1_x0 / integral_sigma2;
        else
            b_target(:, i) = zeros(3, 1);
        end
    end
end

function [loss, gradients] = modelLossU(net, inputs, targets, Xt, sigma)
    % Compute controller Loss
    u_pred = forward(net, inputs);
    u_proj = u_pred - sum(u_pred .* Xt, 1) .* Xt;

    % empirical mean loss calculation
    errors = u_proj + sigma * targets;
    loss = mean(sum(errors.^2, 1));

    % Extract analytical parameter gradients (automatic differentiation)
    gradients = dlgradient(loss, net.Learnables);
end

function [loss, gradients] = modelLossH(net, inputs, targets)
    % Compute corrector Loss
    h_pred = forward(net, inputs);
    h_proj = h_pred - sum(h_pred .* inputs, 1) .* inputs;

    % empirical mean loss calculation
    errors = h_proj - targets;
    loss = mean(sum(errors.^2, 1));

    % Extract analytical parameter gradients (automatic differentiation)
    gradients = dlgradient(loss, net.Learnables);
end