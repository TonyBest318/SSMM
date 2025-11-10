function stifin()
clear; clc; close all;

set(0, 'DefaultAxesFontName', 'Times New Roman')
set(0, 'DefaultTextFontName', 'Times New Roman')
set(0, 'DefaultLegendFontName', 'Times New Roman') % for legends
set(0, 'DefaultColorbarFontName', 'Times New Roman') % for colorbars

p = defaultParams();
disp(p);

% checkpoint parameters
current_time = datetime('now');
save_path = strcat('run_logs/', string(current_time, 'yyyyMMdd_HHmmss'), '_bcf_skt_', sprintf('%.2f', p.f));
mkdir(save_path);
save(strcat(save_path, '/params.mat'), 'p');

% Initial states and control
q0 = paddle_kinematics([0.0; 0.0; 0.0; 0.0], p.L);
x0 = [q0; zeros(p.n_v + p.n_lambda, 1)];
u0 = [0.3; 0.2; 0.05];    % stiffness K hold constant in this simulation
% u0 = [0.02; 0.02; 0.02];

theta_all = zeros(p.n_steps, 4);      % All link angles for visual
theta_all(1,1) = x0(3, 1);
theta_all(1,2) = x0(6, 1);
theta_all(1,3) = x0(9, 1);
theta_all(1,4) = x0(12, 1);

forces_y_all = zeros(p.n_steps, 1);   % All thrust forces for visual
% forces_y_all(1,1) = calculate_physics_thrust(x0, p);

% initial sequence yk and uk for optimization
y_init = zeros(p.n_steps, p.n_x);
y_init(1, :) = x0;
K_traj_initial = initialStiffness(u0, p);
Kx = open('run_logs/20250825_120612_bcf_skt/x_it_02_0.271614.mat').x;
K_traj_initial = reshape(Kx(p.n_colloc*p.n_x+1:end), 1, p.N_total);

xk = x0;
% Main simulation loop
for i = 2:p.n_steps
    cur_t = (i-2)*p.dt;
    uk = interp_stiffness(cur_t, p.t_control, K_traj_initial);
    xkm1 = xk;
    xk = dynamics_step(xk, uk, cur_t, p);
    % 修正：直接使用绝对角度
    theta_all(i,1) = xk(3, 1);
    theta_all(i,2) = xk(6, 1);
    theta_all(i,3) = xk(9, 1);
    theta_all(i,4) = xk(12, 1);

    forces_y_all(i, 1) = calculate_bcf_thrust_x(xk, xkm1, p, p.dt);
    % update for initial sequence
    y_init(i, :) = xk;
end
disp(num2str(mean(forces_y_all), 6));
disp(num2str(mean(forces_y_all(1:500)), 6));
disp(num2str(mean(forces_y_all(501:end)), 6));

% Visualization
% figure('Position', [100, 100, 400, 400]);

% subplot(4,1,1);
% plot_thrust(forces_y_all, theta_all, p);


% subplot(4,1,2);
% plot_Kt_func(Kx, p);

% save_force_csv(forces_y_all, p);

% 生成动画和静态图
% appendage_animation(theta_all, uk, p);
% create_cycle_overlay_figure(theta_all, uk, p);

% Construct decision vector for optimization
% Format: [states at collocation points; stiffness values]
y_colloc = y_init(p.colloc_indices, :);
decision_vars = [reshape(y_colloc', [], 1); reshape(K_traj_initial, [], 1)];

% Setup bounds for decision variables
lb = -Inf*ones(size(decision_vars));
ub = Inf*ones(size(decision_vars));

% state bounds (q, v, lambda)
% joint max velocity
% stiffness bounds
for i=1:p.N_total
    idxk = p.n_x*p.n_colloc+(i-1);
    for j=1:1
        lb(idxk+j) = p.K_min(j);
        ub(idxk+j) = p.K_max(j);
    end
end

% Linear eq constr: Initial state constraint
% also Kt symmetric constraint
n_sym_constrs = (p.N_u/2*p.n_cycles-1);
Aeq = zeros(p.n_x+n_sym_constrs, length(decision_vars));
beq = zeros(p.n_x+n_sym_constrs, 1);
% Set initial state to x0
for i = 1:p.n_x
    Aeq(i, i) = 1;
    beq(i) = x0(i, 1);
end
for cycleid = 1:p.n_cycles
    for cnstrid = 1:p.N_u/2
        if (cycleid==p.n_cycles) && (cnstrid==p.N_u/2)
            continue
        end
        cidx=p.N_u/2*(cycleid-1)+cnstrid;
        kid1=p.N_u/2+1-cnstrid;
        kid2=p.N_u+2-kid1;
        kid1=kid1+p.N_u*(cycleid-1);
        kid2=kid2+p.N_u*(cycleid-1);
        for j=1:1
            cidxj=(cidx-1)+j+p.n_x;
            kid1j=p.n_x*p.n_colloc+(kid1-1)+j;
            kid2j=p.n_x*p.n_colloc+(kid2-1)+j;
            Aeq(cidxj, kid1j) = 1.0;
            Aeq(cidxj, kid2j) = -1.0;
            beq(cidxj) = 0.0;
        end
    end
end

% Linear ineq constr: inequality for stiffness range
Aineq = zeros(2*p.N_total, length(decision_vars));
bineq = zeros(2*p.N_total, 1);
for i = 1:1
    for j = 1:p.N_total
        idx = p.n_x*p.n_colloc + i + (j-1);
        nidx = idx+1;
        if j == p.N_total
            nidx = p.n_x*p.n_colloc + i;
        end
        rid = (i-1)*2*p.N_total+2*j-1;
        % mtd1: by difference
        Aineq(rid, idx) = 1;
        Aineq(rid, nidx) = -1;
        bineq(rid) = 0.1;
        Aineq(rid+1, idx) = -1;
        Aineq(rid+1, nidx) = 1;
        bineq(rid+1) = 0.1;
        % mtd2: by ratio
        % Aineq(rid, idx) = 0.9;
        % Aineq(rid, nidx) = -1.0;
        % bineq(rid) = 0.0;
        % Aineq(rid+1, idx) = -1.1;
        % Aineq(rid+1, nidx) = 1.0;
        % bineq(rid+1) = 0.0;
    end
end

% Setup nonlinear constraint function for dynamics
nonlcon = @(x) dynamics_constraints(x, p);

% Setup objective function
objfun = @(x) thrust_objective(x, p);

outfun = @(x, optimValues, state) optOutfun(x, optimValues, state, save_path);

% Optimization options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 1000000, 'MaxIterations', 140, 'UseParallel', true, ...
    'SubproblemAlgorithm', 'cg', 'ScaleProblem', false, 'HonorBounds', true, ...
    OutputFcn=outfun);

% Run optimization
fprintf('Starting optimization...\n');
[x_opt, fval] = fmincon(objfun, decision_vars, Aineq, bineq, Aeq, beq, lb, ub, nonlcon, options);

% Extract optimized stiffness
% K_traj_opt = reshape(x_opt(6*n_colloc+1:end), 3, N_total);

% Extract optimized states at collocation points
% y_opt_colloc = reshape(x_opt(1:6*n_colloc), 6, n_colloc)';


end


%% ============ Optimization ============
function [c, ceq] = dynamics_constraints(x, p)
    % No inequality constraints
    c = [];
    
    % Extract states and controls from decision vector
    states = reshape(x(1:p.n_colloc*p.n_x), p.n_x, p.n_colloc)';
    K_traj = reshape(x(p.n_colloc*p.n_x+1:end), 1, p.N_total);

    % get size of dynamics residual first
    u_k = interp_stiffness(0.0, p.t_control, K_traj);
    r0 = complete_dynamics_residual(states(2, :)', states(1, :)', u_k, 0.0, p.dt_colloc, p);
    Nr = length(r0);
    % Initialize equality constraints (dynamics continuity)
    ceq = zeros(p.n_colloc-1, Nr);
    ceq(1, :) = r0;

    states_p1 = zeros(p.n_colloc, p.n_x);
    states_p1(2:end-1, :) = states(3:end, :);
    parfor i=2:p.n_colloc-1
        t = (i-2)*p.dt_colloc;
        x_kp1 = states_p1(i, :)';
        x_k = states(i, :)';
        u_k = interp_stiffness(t, p.t_control, K_traj);
        rk = complete_dynamics_residual(x_kp1, x_k, u_k, t, p.dt_colloc, p);
        ceq(i, :) = rk;
    end
    ceq = reshape(ceq', (p.n_colloc-1)*Nr, 1);
end

function obj = thrust_objective(x, p)
    % Extract states and controls from decision vector
    states = reshape(x(1:p.n_colloc*p.n_x), p.n_x, p.n_colloc)';

    forces_y_all = zeros(p.n_colloc, 1);   % All thrust forces
    for i=1:p.n_colloc
        x_k = states(i, :)';
        x_km1 = states(max(1, i-1), :)';
        forces_y_all(i, 1) = calculate_bcf_thrust_x(x_k, x_km1, p, p.dt_colloc);
    end
    avg_thrust = mean(forces_y_all);

    % objective minimize negative mean force
    obj = -avg_thrust * 1e5;
end

function stop = optOutfun(x, optimValues, state, save_path)
    stop = false;

    fname = sprintf('/x_it_%02d_%.6f.mat', optimValues.iteration, -optimValues.fval/1e5);
    save(strcat(save_path, fname), 'x');
end


%% ============ Parameters ============
function p = defaultParams()
    p.f    = 1.0;        % Hz
    p.A    = 0.52;        % rad
    p.phi0 = 0;
    p.bias = deg2rad(0);
    
    p.n_links = 4;
    p.n_q = 3*p.n_links;
    p.n_v = 3*p.n_links;
    p.n_lambda = 2*p.n_links+1;
    p.n_x = p.n_q + p.n_v + p.n_lambda;

    % Link lengths (m)
    p.l1 = 0.125;
    p.l2 = 0.032;
    p.l3 = 0.032;
    p.l4 = 0.032;
    p.L = [p.l1; p.l2; p.l3; p.l4];
    
    % Center of mass distances (m) - adjusted for effective hydrodynamic portions
    p.lc1 = 0.5*p.l1;  % COM for effective hydrodynamic portion of L1
    p.lc2 = 0.5*p.l2;
    p.lc3 = 0.5*p.l3;
    p.lc4 = 0.5*p.l4;
    
    % Link widths (m)
    p.w1 = 0.05;  
    p.w2 = 0.05;  
    p.w3 = 0.05;  
    p.w4 = 0.05;
    p.w = [p.w1; p.w2; p.w3; p.w4];
    
    % Link thickness (m)
    p.h = 0.004;
    
    % Link masses (kg) - adjusted for hydrodynamic portions
    p.m1 = 0.04;
    p.m2 = 0.3*p.l2;
    p.m3 = 0.3*p.l3;
    p.m4 = 0.3*p.l4;
    
    % Link moments of inertia (kg?m?)
    p.J1 = (1/12)*p.m1*(p.l1^2);  % Effective inertia of L1 for hydrodynamics
    p.J2 = (1/12)*p.m2*(p.l2^2);
    p.J3 = (1/12)*p.m3*(p.l3^2);
    p.J4 = (1/12)*p.m4*(p.l4^2);

    p.M = mass_matrix([p.m1; p.m2; p.m3; p.m4], [p.J1; p.J2; p.J3; p.J4]);
    
    % Hydrodynamic parameters
    p.rho = 1000;            % Water density (kg/m?)
    p.Ca   = 1.0;   % 附加质量系数（乘在 π/4 上）
    p.Cn   = 2.6;   % 法向阻力
    p.Ct   = 0.2;   % 切向阻力
    p.Crot = 0.1;   % 旋转阻力矩
    
    % Joint stiffness parameters boundaries (N?m/rad)
    p.K_min = [0.00001; 0.00001; 0.00001];  % Minimum stiffness values
    p.K_max = [0.35; 0.35; 0.35];     % Maximum stiffness values

    % Number of control points for stiffness trajectory (per cycle)
    p.N_u = 10;  % Number of control points per cycle
    p.t_final = 1/p.f*2;  % Simulation end time (s)
    p.dt = 1/p.f*0.002;  % Time step (s)
    p.t_sol = 0:p.dt:p.t_final;
    p.n_steps = length(p.t_sol);

    % collocation settings
    p.colloc_step = 5;
    p.colloc_indices = 1:p.colloc_step:p.n_steps;
    p.n_colloc = length(p.colloc_indices);
    p.dt_colloc = p.dt * p.colloc_step;
    
    % Time points for stiffness control (one complete cycle)
    p.cycle_time = 1/p.f;
    p.t_control_cycle = linspace(0, p.cycle_time, p.N_u+1);
    p.t_control_cycle = p.t_control_cycle(1:end-1);  % Remove duplicate at cycle boundary
    
    % Repeat pattern for full simulation duration
    p.n_cycles = ceil(p.t_final / p.cycle_time);
    p.t_control = [];
    for i = 0:p.n_cycles-1
        p.t_control = [p.t_control, p.t_control_cycle + i*p.cycle_time];
    end
    p.t_control = p.t_control(p.t_control <= p.t_final);
    p.N_total = length(p.t_control);
end

%% ============ Color and Styling Functions ============
function color = get_stiffness_color(stiffness_value, K_min, K_max)
    % 根据刚度值计算颜色，从黄色(低)到红色(高)
    % 归一化刚度值到[0,1]
    normalized_stiffness = (stiffness_value - K_min) / (K_max - K_min);
    normalized_stiffness = max(0, min(1, normalized_stiffness)); % 确保在[0,1]范围内
    
    % 黄色 [1, 1, 0] 到 红色 [1, 0, 0]
    color = [1, 1-normalized_stiffness, 0];
end

function marker_size = get_stiffness_marker_size(stiffness_value, K_min, K_max, base_size, size_range)
    % 根据刚度值计算关节大小，使用对数缩放来避免过度变化
    % 归一化刚度值到[0,1]
    normalized_stiffness = (stiffness_value - K_min) / (K_max - K_min);
    normalized_stiffness = max(0, min(1, normalized_stiffness));
    
    % 使用对数缩放，让变化更平滑，类似肌肉收缩的感觉
    % 应用sqrt函数让变化更温和，避免过度夸张
    size_factor = sqrt(normalized_stiffness);
    
    % 计算最终大小
    marker_size = base_size + size_range * size_factor;
end

function create_cycle_overlay_figure(theta_all, uk, p)
    % 创建一个周期内8个状态的叠加图
    figure('Position', [200, 200, 800, 600], 'Name', 'One Cycle Overlay', 'Color', 'k');
    
    % 计算一个周期的步数
    cycle_steps = round(p.cycle_time / p.dt);
    
    % 选择8个时间点 (T/8 间隔)
    n_snapshots = 8;
    snapshot_indices = round(linspace(1, cycle_steps, n_snapshots));
    
    % 透明度从高到低
    alphas = linspace(0.9, 0.3, n_snapshots);
    
    % 设置坐标轴和背景
    ax = gca;
    set(ax, 'Color', 'k');  % 设置坐标轴背景为黑色
    hold on;
    
    % 绘制每个快照
    for snap_idx = 1:n_snapshots
        i = snapshot_indices(snap_idx);
        alpha = alphas(snap_idx);
        
        % 使用绝对角度
        th1 = theta_all(i,1);
        th2 = theta_all(i,2);
        th3 = theta_all(i,3);
        th4 = theta_all(i,4);
        
        % 计算链节位置
        p_servo = [0; 0];
        p1 = p_servo + p.l1 * [cos(th1); sin(th1)];
        p2 = p1 + p.l2 * [cos(th2); sin(th2)];
        p3 = p2 + p.l3 * [cos(th3); sin(th3)];
        p4 = p3 + p.l4 * [cos(th4); sin(th4)];
        
        % 绘制白色半透明链节 - 使用正确的透明度语法
        h_link = plot([p_servo(1), p1(1), p2(1), p3(1), p4(1)], ...
                     [p_servo(2), p1(2), p2(2), p3(2), p4(2)], ...
                     'Color', [1, 1, 1], 'LineWidth', 4);
        % 设置透明度
        h_link.Color(4) = alpha;
        
        % 绘制渐变色关节 - 只有真正的关节，不包括末端
        joint_positions = [p_servo, p1, p2, p3, p4];
        
        % 基座关节 (固定颜色和大小)
        plot(joint_positions(1,1), joint_positions(2,1), 'o', ...
             'Color', [0.8, 0.8, 0.8], 'MarkerSize', 8, 'MarkerFaceColor', [0.8, 0.8, 0.8]);
        
        % 关节1,2,3 (基于刚度的渐变色和大小)
        for joint_idx = 2:4  % 只画关节1,2,3，不包括末端点
            stiffness_idx = joint_idx - 1;
            joint_color = get_stiffness_color(uk(stiffness_idx), ...
                                            p.K_min(stiffness_idx), p.K_max(stiffness_idx));
            joint_size = get_stiffness_marker_size(uk(stiffness_idx), ...
                                                  p.K_min(stiffness_idx), p.K_max(stiffness_idx), ...
                                                  6, 8);  % 基础大小6，变化范围8
            
            plot(joint_positions(1,joint_idx), joint_positions(2,joint_idx), 'o', ...
                 'Color', joint_color, 'MarkerSize', joint_size, 'MarkerFaceColor', joint_color);
        end
        
        % 末端点 (小的白色点，表示末端位置，不是关节)
        plot(joint_positions(1,5), joint_positions(2,5), 'o', ...
             'Color', [1, 1, 1], 'MarkerSize', 3, 'MarkerFaceColor', [1, 1, 1]);
    end
    
    % 设置坐标轴和标题
    axis equal;
    xlim([-0.25, 0.3]);
    ylim([-0.20, 0.3]);
    xlabel('x (m)', 'Color', 'w', 'FontSize', 12);
    ylabel('y (m)', 'Color', 'w', 'FontSize', 12);
    title('One Cycle Motion Overlay (T/8 intervals)', 'Color', 'w', 'FontSize', 14);
    
    % 设置坐标轴颜色
    ax = gca;
    ax.XColor = 'w';
    ax.YColor = 'w';
    ax.GridColor = [0.3, 0.3, 0.3];
    grid on;
    
    % 添加颜色条说明
    text(-0.22, 0.25, 'Joint Stiffness', 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    text(-0.22, 0.22, 'Yellow → Red', 'Color', 'w', 'FontSize', 9);
    text(-0.22, 0.19, '(Low → High)', 'Color', 'w', 'FontSize', 9);
    
    % 添加透明度说明
    text(0.15, 0.25, 'Transparency', 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    text(0.15, 0.22, 'High → Low', 'Color', 'w', 'FontSize', 9);
    text(0.15, 0.19, '(Early → Late)', 'Color', 'w', 'FontSize', 9);
end

%% ============ Visualization ============
function appendage_animation(theta_all, uk, p)
    % 创建动画窗口 - 黑色背景
    anim_fig = figure('Position', [100, 100, 800, 600], 'Name', 'Appendage Animation', 'Color', 'k');
    
    n_steps = p.n_steps;
    % 计算末端执行器轨迹
    p_end = zeros(n_steps, 2);
    
    % 预计算所有位置
    p_servo_all = zeros(n_steps, 2);
    p1_all = zeros(n_steps, 2);
    p2_all = zeros(n_steps, 2);
    p3_all = zeros(n_steps, 2);
    p4_all = zeros(n_steps, 2);
    
    for i = 1:n_steps
        % 使用绝对角度
        th1 = theta_all(i,1);
        th2 = theta_all(i,2);
        th3 = theta_all(i,3);
        th4 = theta_all(i,4);
        
        % 计算链节位置
        p_servo = [0; 0];
        p1 = p_servo + p.l1 * [cos(th1); sin(th1)];
        p2 = p1 + p.l2 * [cos(th2); sin(th2)];
        p3 = p2 + p.l3 * [cos(th3); sin(th3)];
        p4 = p3 + p.l4 * [cos(th4); sin(th4)];
        
        % 存储所有位置
        p_servo_all(i,:) = p_servo';
        p1_all(i,:) = p1';
        p2_all(i,:) = p2';
        p3_all(i,:) = p3';
        p4_all(i,:) = p4';
        
        p_end(i,:) = p4';
    end
    
    % 动画设置
    figure(anim_fig);
    
    % 设置坐标轴背景为黑色
    ax = gca;
    set(ax, 'Color', 'k');  % 坐标轴背景黑色
    hold on;
    
    % 创建图形对象 - 白色链节，更粗的线条
    h_appendage = plot([0 0 0 0 0], [0 0 0 0 0], 'Color', [1, 1, 1], 'LineWidth', 4);
    
    % 创建渐变色关节
    h_joint_base = plot(0, 0, 'o', 'Color', [0.8, 0.8, 0.8], 'MarkerSize', 8, 'MarkerFaceColor', [0.8, 0.8, 0.8]);
    h_joint1 = plot(0, 0, 'o', 'MarkerSize', 8);
    h_joint2 = plot(0, 0, 'o', 'MarkerSize', 8);
    h_joint3 = plot(0, 0, 'o', 'MarkerSize', 8);
    h_end_point = plot(0, 0, 'o', 'Color', [1, 1, 1], 'MarkerSize', 4, 'MarkerFaceColor', [1, 1, 1]);  % 末端点
    
    h_trace = plot(0, 0, 'Color', [0.7, 0.7, 0.7], 'LineStyle', ':', 'LineWidth', 2);
    h_phase = text(0.15, 0.15, '', 'FontSize', 12, 'Color', 'w');
    h_note = text(-0.2, 0.25, sprintf('Link Properties - L1: %.1fcm, L2-4: %.1fcm', ...
                100*p.l1, 100*p.l2), 'FontSize', 9, 'Color', 'w');
    
    % 设置坐标轴
    axis equal;
    xlim([-0.25, 0.3]);
    ylim([-0.20, 0.3]);
    xlabel('x (m)', 'Color', 'w', 'FontSize', 12);
    ylabel('y (m)', 'Color', 'w', 'FontSize', 12);
    title(sprintf('Flexible Appendage Animation'), 'Color', 'w', 'FontSize', 14);
    
    % 设置坐标轴颜色
    ax = gca;
    ax.XColor = 'w';
    ax.YColor = 'w';
    ax.GridColor = [0.3, 0.3, 0.3];
    grid on;
    
    % 预计算关节颜色和大小
    joint_colors = zeros(3, 3); % 3个关节，RGB颜色
    joint_sizes = zeros(3, 1);  % 3个关节大小
    for i = 1:3
        joint_colors(i,:) = get_stiffness_color(uk(i), p.K_min(i), p.K_max(i));
        joint_sizes(i) = get_stiffness_marker_size(uk(i), p.K_min(i), p.K_max(i), 6, 8);
    end
    
    % 动画循环
    frame_skip = 10;
    for i = 1:frame_skip:n_steps
        % 检查对象是否仍然有效
        if ~ishandle(h_appendage) || ~ishandle(h_joint1)
            fprintf('Animation objects are no longer valid. Stopping animation.\n');
            break;
        end
        
        % 获取预计算的位置
        p_servo = p_servo_all(i,:)';
        p1 = p1_all(i,:)';
        p2 = p2_all(i,:)';
        p3 = p3_all(i,:)';
        p4 = p4_all(i,:)';
        
        % 更新附肢段 - 白色，更粗
        set(h_appendage, 'XData', [p_servo(1), p1(1), p2(1), p3(1), p4(1)], ...
                        'YData', [p_servo(2), p1(2), p2(2), p3(2), p4(2)]);
        
        % 更新关节 - 渐变色和大小
        set(h_joint_base, 'XData', p_servo(1), 'YData', p_servo(2));
        
        set(h_joint1, 'XData', p1(1), 'YData', p1(2), ...
                     'Color', joint_colors(1,:), 'MarkerFaceColor', joint_colors(1,:), ...
                     'MarkerSize', joint_sizes(1));
        set(h_joint2, 'XData', p2(1), 'YData', p2(2), ...
                     'Color', joint_colors(2,:), 'MarkerFaceColor', joint_colors(2,:), ...
                     'MarkerSize', joint_sizes(2));
        set(h_joint3, 'XData', p3(1), 'YData', p3(2), ...
                     'Color', joint_colors(3,:), 'MarkerFaceColor', joint_colors(3,:), ...
                     'MarkerSize', joint_sizes(3));
        
        % 更新末端点 - 小的白色点
        set(h_end_point, 'XData', p4(1), 'YData', p4(2));
        
        % 更新轨迹
        set(h_trace, 'XData', p_end(1:i,1), 'YData', p_end(1:i,2));
        
        % 确定行程阶段
        % [~, is_power_stroke, ~] = asymmetric_driving_signal((i-1)*p.dt, p);
        is_power_stroke = true;
        if is_power_stroke
            stroke_label = 'Power Stroke';
            set(h_phase, 'String', stroke_label, 'Color', 'r');
        else
            stroke_label = 'Recovery Stroke';
            set(h_phase, 'String', stroke_label, 'Color', [0.5, 0.5, 1]);
        end
        
        % 添加时间和刚度信息
        title(sprintf('Time: %.2fs - %s', (i-1)*p.dt, stroke_label), 'Color', 'w', 'FontSize', 14);
        
        drawnow;
        pause(p.dt*frame_skip);
    end
    
    % 添加颜色条说明
    text(-0.22, 0.15, 'Joint Properties:', 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
    text(-0.22, 0.12, 'Color: Yellow → Red (Low → High Stiffness)', 'Color', 'w', 'FontSize', 9);
    text(-0.22, 0.09, 'Size: Small → Large (Low → High Stiffness)', 'Color', 'w', 'FontSize', 9);
    for i = 1:3
        color_str = sprintf('Joint %d: K=%.3f, Size=%.1f', i, uk(i), joint_sizes(i));
        text(-0.22, 0.06-i*0.02, color_str, 'Color', joint_colors(i,:), 'FontSize', 9);
    end
    text(-0.22, -0.02, 'End Point: White dot (no joint)', 'Color', 'w', 'FontSize', 9);
end

function plot_thrust(thrust_y, theta_all, p)
    % 绘制总推力
    plot(p.t_sol, thrust_y, 'LineWidth', 1.5);
    xlim([0.0, 1.0/p.f]);
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Thrust Force (N)', 'FontSize', 10);
    % title('Total Thrust Force in Y-Direction', 'FontSize', 12);
    % grid on;
    
    % 添加平均推力线
    avg_thrust = mean(thrust_y);
    hold on;
    plot([0, p.t_final], [avg_thrust, avg_thrust], 'r--', 'LineWidth', 1);
    % legend('Thrust', sprintf('Average = %.3f N', avg_thrust), 'Location', 'best');
    hold off;

    yyaxis right;
    ax = gca; % get current axes
    ax.YColor = 'k'; % 'k' = black
    plot(p.t_sol, rad2deg(theta_all(:,1)), 'g-.', 'LineWidth', 1.5);
    ylabel('Angle (deg)', 'FontSize', 10);
    % title('Joint Angles Evolution');
    legend('Thrust', sprintf('Average = %.3f N', avg_thrust), 'θ1', 'Location', 'best');
    grid on;
end

function plot_angle_evolution(theta_all, p)
    % 绘制角度演化
    plot(p.t_sol, rad2deg(theta_all(:,1)), 'LineWidth', 1.5);
    hold on;
    plot(p.t_sol, rad2deg(theta_all(:,2)-theta_all(:,1)), 'LineWidth', 1.5);
    plot(p.t_sol, rad2deg(theta_all(:,3)-theta_all(:,2)), 'LineWidth', 1.5);
    plot(p.t_sol, rad2deg(theta_all(:,4)-theta_all(:,3)), 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Angle (deg)');
    title('Joint Angles Evolution');
    legend('θ?', 'θ?-θ?', 'θ?-θ?', 'θ?-θ?', 'Location', 'best');
    grid on;
end

function plot_Kt_func(x, p)
    K_traj_opt = reshape(x(p.n_colloc*p.n_x+1:end), p.n_links-1, p.N_total);
    
    t_base  = p.t_control;         
    K_base  = K_traj_opt;
    
    tc = [t_base, p.t_final];         
    K1 = [K_base(1,:), K_base(1)];      % Joint 2
    K2 = [K_base(2,:), K_base(2)];      % Joint 3
    K3 = [K_base(3,:), K_base(3)];      % Joint 4
    % K1 = K_base(1,:);      % Joint 2
    % K2 = K_base(2,:);      % Joint 3
    % K3 = K_base(3,:);      % Joint 4
    
    [tc, idx] = sort(tc);
    K1 = K1(idx);
    K2 = K2(idx);
    K3 = K3(idx);
    
    t_fine = linspace(0, p.t_final, 1000);
    
    K1_fine = interp1(tc, K1, t_fine, 'pchip', 'extrap');
    K2_fine = interp1(tc, K2, t_fine, 'pchip', 'extrap');
    K3_fine = interp1(tc, K3, t_fine, 'pchip', 'extrap');
    
    % figure('Position',[100,100,900,400]);
    figure('Position', [100, 100, 400, 200]);
    hold on;
    plot(t_fine, K1_fine, '-', 'LineWidth',1.5);
    plot(t_base, K_base(1,:), 'o','MarkerSize',6);
    hold off;
    xlim([0.0, 1.0/p.f]);
    xlabel('Time (s)');
    ylabel('Stiffness (Nm/rad)');
    % title('Continuous stiffness profile after control point + interpolation');
    % legend('Joint2 cont', ...
    %        'Joint2 pts','Location','best');
    grid on;
    hold off;
    % hold on;
    figure('Position', [100, 100, 400, 200]);
    hold on;
    plot(t_fine, K2_fine, '-', 'LineWidth',1.5);
    plot(t_base, K_base(2,:), 'o','MarkerSize',6);
    hold off;
    xlim([0.0, 1.0/p.f]);
    xlabel('Time (s)');
    ylabel('Stiffness (Nm/rad)');
    % title('Continuous stiffness profile after control point + interpolation');
    % legend('Joint3 cont', ...
    %        'Joint3 pts','Location','best');
    grid on;
    hold off;
    figure('Position', [100, 100, 400, 200]);
    hold on;
    plot(t_fine, K3_fine, '-', 'LineWidth',1.5);
    plot(t_base, K_base(3,:), 'o','MarkerSize',6);
    hold off;
    % xlim([0.0, 1.0]);
    xlim([0.0, 1.0/p.f]);
    xlabel('Time (s)');
    ylabel('Stiffness (Nm/rad)');
    % title('Continuous stiffness profile after control point + interpolation');
    % legend('Joint4 cont', ...
    %        'Joint4 pts','Location','best');
    grid on;
    hold off;
end


%% ================== BCF Hydrodynamics ==================
function M_a = added_mass_matrix_bcf(q, p)
    n = p.n_links; M_a = zeros(3*n, 3*n);
    for i = 1:n
        idx = 3*i-2; theta = q(idx+2); l = p.L(i); w = p.w(i);
        m_a = p.Ca * p.rho * (pi/4) * (w^2) * l;    % 横向附加质量
        I_a = m_a * l^2 / 12;
        c = cos(theta); s = sin(theta);
        n_hat = [-s; c];
        M_xy = m_a * (n_hat * n_hat.');            % 2x2（仅法向）
        M_a(idx:idx+1, idx:idx+1) = M_a(idx:idx+1, idx:idx+1) + M_xy;
        M_a(idx+2, idx+2) = M_a(idx+2, idx+2) + I_a;
    end
end

function F = hydrodynamic_resistive_bcf(q, v, p)
    n = p.n_links; F = zeros(size(v));
    for i = 1:n
        idx = 3*i-2;
        vx = v(idx); vy = v(idx+1); omega = v(idx+2);
        theta = q(idx+2); l = p.L(i); w = p.w(i);
        c = cos(theta); s = sin(theta);
        t_hat = [ c; s]; n_hat = [-s; c];
        v_vec = [vx; vy];
        v_t = dot(v_vec, t_hat); v_n = dot(v_vec, n_hat);
        Ft = -0.5 * p.rho * p.Ct * l * w * abs(v_t) * v_t;
        Fn = -0.5 * p.rho * p.Cn * l * w * abs(v_n) * v_n;
        F_xy = Ft * t_hat + Fn * n_hat;
        Tz   = -0.5 * p.rho * p.Crot * (l^3 * w / 12) * abs(omega) * omega;
        F(idx)   = F(idx)   + F_xy(1);
        F(idx+1) = F(idx+1) + F_xy(2);
        F(idx+2) = F(idx+2) + Tz;
    end
end

function total_thrust_x = calculate_bcf_thrust_x(x_curr, x_prev, p, dt)
% 对"流体"的 x 向推力 = (-F_drag_on_body)_x + d/dt(M_a v)_x
    n_q = p.n_q; n_v = p.n_v;

    qk  = x_prev(1:n_q);   qkp = x_curr(1:n_q);
    vk  = x_prev(n_q+1:n_q+n_v);
    vkp = x_curr(n_q+1:n_q+n_v);

    % —— 阻力：取中点
    q_mid = 0.5*(qkp + qk);
    v_mid = 0.5*(vkp + vk);
    F_res_body  = hydrodynamic_resistive_bcf(q_mid, v_mid, p);   % 在"体"上
    F_res_fluid = -F_res_body;                                   % 在"流体"上

    % —— 反应式：用附加动量的差分（包含 \dot{M}_a v 项）
    Ma_k  = added_mass_matrix_bcf(qk,  p);
    Ma_kp = added_mass_matrix_bcf(qkp, p);
    F_react_fluid = (Ma_kp*vkp - Ma_k*vk) / dt;

    % —— x 分量求和
    total_thrust_x = 0.0;
    for i = 1:p.n_links
        idx = 3*i-2;  
        total_thrust_x = total_thrust_x + F_res_fluid(idx) + F_react_fluid(idx);
    end
end


%% ============ Dynamics ============
function residual = q_residual(x_kp1, x_k, dt, p)
    q_kp1 = x_kp1(1:p.n_q);
    q_k = x_k(1:p.n_q);
    v_kp1 = x_kp1(p.n_q+1:p.n_q+p.n_v);
    residual = q_kp1 - q_k - v_kp1*dt;
end

function residual = body_constraint_residual(q, t, p)
    L = p.L;
    % hinge constraint (link 1 to base)
    r1 = q(1:2);
    theta1 = q(3);
    l1 = L(1);
    hinge_constraint_residual = r1 - [l1/2*cos(theta1); l1/2*sin(theta1)];

    [theta1_ref, ~] = driving_signal_bcf(t, p);
    driving_constraint_residual = theta1 - theta1_ref;
    
    joint_constraint_residual = zeros(2*(length(L)-1), 1);
    
    for i = 1:(length(L)-1)
        q_i = q(3*(i-1)+1:3*i);
        r_i = q_i(1:2);
        theta_i = q_i(3);
        L_i = L(i);
        
        q_ip1 = q(3*i+1:3*(i+1));
        r_ip1 = q_ip1(1:2);
        theta_ip1 = q_ip1(3);
        L_ip1 = L(i+1);
        
        joint_constraint_residual(2*(i-1)+1:2*i) = r_i + [L_i/2*cos(theta_i); L_i/2*sin(theta_i)] - ...
            (r_ip1 - [L_ip1/2*cos(theta_ip1); L_ip1/2*sin(theta_ip1)]);
    end
    
    residual = [hinge_constraint_residual; driving_constraint_residual; joint_constraint_residual];
end

function J = body_constraint_jacobian(q, t, p)
    % Using finite difference utility function
    func = @(x) body_constraint_residual(x, t, p);
    J = finite_difference_jacobian(func, q);
end

function K = stiffness_matrix(k)
    difference_matrix = zeros(length(k), 3*(length(k)+1));
    
    for i = 1:length(k)
        ind_i = (i-1)*3 + 3;
        ind_i_plus_1 = i*3 + 3;
        difference_matrix(i, ind_i) = -1.0;
        difference_matrix(i, ind_i_plus_1) = 1.0;
    end
    
    K = difference_matrix' * diag(k) * difference_matrix;
end

function grad = potential_grad(q, K)
    grad = K * q;
end

function residual = stationarity_residual(x_kp1, x_k, u_k, t, dt, p)
    M = p.M;
    n_q = p.n_q;
    n_v = p.n_v;

    q_kp1 = x_kp1(1:n_q);
    q_k = x_k(1:n_q);
    v_kp1 = x_kp1(n_q+1:n_q+n_v);
    v_k = x_k(n_q+1:n_q+n_v);
    
    K_k = stiffness_matrix(u_k);
    lambda_k = x_kp1(n_q+n_v+1:end);
    dc_dq_k = body_constraint_jacobian(q_k, t, p);
    
    % Midpoint rule for better stability
    q_mid = q_k + 0.5*dt*v_kp1;
    M_a_k   = added_mass_matrix_bcf(q_k,   p);
    M_a_mid = added_mass_matrix_bcf(q_mid, p);
    
    stationarity_residual_kp1 = (M + M_a_mid)*v_kp1 + (dt/2)*potential_grad(q_mid, K_k) - ...
        dt*dc_dq_k'*lambda_k - (dt/2)*hydrodynamic_resistive_bcf(q_kp1, v_kp1, p);
    
    stationarity_residual_k = (M + M_a_k)*v_k - (dt/2)*potential_grad(q_k - 0.5*dt*v_k, K_k) + ...
        (dt/2)*hydrodynamic_resistive_bcf(q_k, v_k, p);
    
    residual = stationarity_residual_kp1 - stationarity_residual_k;
end

function residual = complete_dynamics_residual(x_kp1, x_k, u_k, t, dt, p)
    % Main function for DCOL solver constraint
    q_kp1 = x_kp1(1:p.n_q);
    
    residual = [stationarity_residual(x_kp1, x_k, u_k, t, dt, p);
                q_residual(x_kp1, x_k, dt, p);
                body_constraint_residual(q_kp1, t+dt, p)];
end

function J = complete_dynamics_jacobian(x_kp1, x_k, u_k, t, dt, p)
    % Using finite difference utility function
    func = @(x) complete_dynamics_residual(x, x_k, u_k, t, dt, p);
    J = finite_difference_jacobian(func, x_kp1);
end

function x_kp1 = dynamics_step(x_k, u_k, t, p)
    % Solves for the trajectory using Newton's method at each time step
    x_kp1 = x_k;  % Initial guess for next state
    dt = p.dt;
    
    residual = complete_dynamics_residual(x_kp1, x_k, u_k, t, dt, p);
    max_iter = 0;
    tol = 1e-10;
    max_iterations = 20;
    
    % Newton's method with line search
    while max(abs(residual)) > tol && max_iter < max_iterations
        % Compute Jacobian
        jacobian = complete_dynamics_jacobian(x_kp1, x_k, u_k, t, dt, p);
        
        % Check condition number
        cond_num = cond(jacobian);
        if cond_num > 1e12
            warning('Jacobian is ill-conditioned. Condition number: %.2e', cond_num);
        end
        
        % Newton step with regularization if needed
        if cond_num > 1e10
            delta_x = -(jacobian + 1e-8*eye(size(jacobian))) \ residual;
        else
            delta_x = -jacobian \ residual;
        end
        
        % Line search for stability
        alpha = 1.0;
        x_test = x_kp1 + alpha * delta_x;
        residual_test = complete_dynamics_residual(x_test, x_k, u_k, t, dt, p);
        
        while norm(residual_test) > norm(residual) && alpha > 1e-4
            alpha = alpha * 0.5;
            x_test = x_kp1 + alpha * delta_x;
            residual_test = complete_dynamics_residual(x_test, x_k, u_k, t, dt, p);
        end
        
        x_kp1 = x_test;
        residual = residual_test;
        
        max_iter = max_iter + 1;
    end
    
    if max_iter >= max_iterations
        warning('Newton method did not converge. Max residual: %.6e', max(abs(residual)));
    end
end

%% ================== Driving (pure sine) ==================
function [theta, dtheta] = driving_signal_bcf(t, p)
    theta  = p.A * sin(2*pi*p.f*t + p.phi0) + p.bias;
    dtheta = 2*pi*p.f * p.A * cos(2*pi*p.f*t + p.phi0);
end

%% ============ Helper Functions ============

function r = paddle_kinematics(theta_joint, L)
    r = zeros(3*length(L), 1);
    r(1:3) = [L(1)/2*cos(theta_joint(1)); L(1)/2*sin(theta_joint(1)); theta_joint(1)];
    
    % 累加角度以获得绝对角度
    cumulative_angle = theta_joint(1);
    
    for i = 2:length(L)
        start_ind = (i-1)*3+1;
        end_ind = i*3;
        r_im1 = r(start_ind-3:end_ind-3);
        
        cumulative_angle = cumulative_angle + theta_joint(i);
        
        r(start_ind:end_ind) = [r_im1(1) + L(i-1)/2*cos(r_im1(3)) + L(i)/2*cos(cumulative_angle);
                                r_im1(2) + L(i-1)/2*sin(r_im1(3)) + L(i)/2*sin(cumulative_angle);
                                cumulative_angle];
    end
end

function M = mass_matrix(m, J)
    M = zeros(3*length(m), 3*length(m));
    
    for i = 1:length(m)
        ind_start = (i-1)*3+1;
        M(ind_start, ind_start) = m(i);
        M(ind_start+1, ind_start+1) = m(i);
        M(ind_start+2, ind_start+2) = J(i);
    end
end

function K_init = initialStiffness(K_nominal, params)
    % Initial stiffness values (to be optimized)
    K_init = zeros(1, params.N_total);
    for i = 1:1
        K_init(i,:) = K_nominal(i);
    end
end

% Interpolate stiffness at any time
function K_effective = interp_stiffness(t, t_control, K_traj)
    % Handle periodic extension for times outside control range
    cycle_time = t_control(end) - t_control(1) + (t_control(2) - t_control(1));
    t_mod = mod(t, cycle_time);
    
    % Interpolate for each joint
    K_effective = zeros(3, 1);
    tc = [t_control, cycle_time];      
    for i = 1:3
        % Use piecewise cubic Hermite interpolation for smoothness
        K1 = [K_traj(1,:), K_traj(1)];
        K_effective(i) = interp1(tc, K1, t_mod, 'pchip', 'extrap');
    end
end

function J = finite_difference_jacobian(func, x, h)
    % Utility function for computing Jacobian using finite differences
    % func: function handle
    % x: point at which to evaluate Jacobian
    % h: step size (optional, default 1e-8)
    
    if nargin < 3
        h = 1e-8;
    end
    
    n = length(x);
    f0 = func(x);
    m = length(f0);
    J = zeros(m, n);
    
    for i = 1:n
        x_plus = x;
        x_plus(i) = x_plus(i) + h;
        f_plus = func(x_plus);
        J(:,i) = (f_plus - f0) / h;
    end
end

function save_force_csv(thrust_y, p)
    % Create table with column names
    data_table = table(p.t_sol', thrust_y, ...
        'VariableNames', {'Time_s', 'Total_Thrust_Y_N'});
    
    % Save to CSV
    current_time = datetime('now');
    filename = sprintf('thrust_y_%s.csv', string(current_time, 'yyyyMMdd_HHmmss'));
    writetable(data_table, filename);
    fprintf('Thrust data saved to: %s\n', filename);
end
