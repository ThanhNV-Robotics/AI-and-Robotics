import numpy as np
import cartPoledynamics
from casadi import *
import yaml

from trajOptUtils import*
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parse yaml config file
    
    with open('trajectory_optimization/cart_pole/env_config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cart_pole = cartPoledynamics.cartPole(config['dynamic_parameters'])

    init_state = np.array([0.0, 0.0, 0.0, 0.0])  # x, theta, dx, dtheta

    # Compute initial guess trajectory
    x_guess, xd_guess, u_guess, t = ComputeGuessSwingUp(config, cart_pole)

    # CasADi: ODE dynamics
    g_val = cart_pole.g
    M     = cart_pole.M
    m     = cart_pole.m
    l     = cart_pole.l

    s_sym = MX.sym('s', 4)   # state [x, theta, dx, dtheta]
    h_sym = MX.sym('h')      # sub-step size
    u_sym = MX.sym('u')      # control input

    x1_s    = s_sym[0] # x
    x2_s    = s_sym[1] # theta
    x3_s    = s_sym[2] # dx
    x4_s    = s_sym[3] # dtheta

    s2 = sin(x2_s)
    c2 = cos(x2_s)

    f1_rhs = x3_s
    f2_rhs = x4_s
    f3_rhs = (u_sym + m * l * x4_s**2 * s2  - m*g_val*c2*s2) / (M + m - m * c2**2)
    f4_rhs = (g_val*(M+m)*s2 - c2*(u_sym+m*l*x4_s**2*s2)) / (l*(M + m - m * c2**2))

    ode_rhs = vertcat(f1_rhs, f2_rhs, f3_rhs, f4_rhs)

    # Wrap ODE as a CasADi Function so it is auto-differentiable
    f_ode = Function('f_ode', [s_sym, u_sym], [ode_rhs])

    # RK4 single-step Function
    k1 = f_ode(s_sym, u_sym)
    k2 = f_ode(s_sym + 0.5*h_sym*k1, u_sym)
    k3 = f_ode(s_sym + 0.5*h_sym*k2, u_sym)
    k4 = f_ode(s_sym + h_sym*k3, u_sym)
    s_next_rk4 = s_sym + (h_sym / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # Integrator function
    F_step = Function('F_step', [s_sym, u_sym, h_sym], [s_next_rk4])

    N= config['multiple_shooting']['N'] # Number of shooting segments
    T= config['multiple_shooting']['T'] # Total time horizon
    nSubSteps = config['multiple_shooting']['nsubsteps'] # Number of sub-steps for integration

    def F_integrate(s, u):
        # Integrate over one shooting segment using RK4 sub-steps
        T_seg = T / N # time duration of each shooting segment
        h = T_seg / nSubSteps
        s_next = s
        for _ in range(nSubSteps):
            s_next = F_step(s_next, u, h)
        return s_next
    # --- NLP decision variables

    s_nodes = [MX.sym(f's_{k}', 4) for k in range(N + 1)]
    u_nodes = [MX.sym(f'u_{k}') for k in range(N)]
    w = vertcat(*s_nodes, *u_nodes)

    # Initial guess: flatten state nodes (column-major) then append control nodes
    w0 = np.concatenate([
    x_guess.flatten(),
    u_guess.flatten()])

    # Lower bound for state
    s_lwb = [-np.inf] * 4 * (N + 1)
    s_uwb = [ np.inf] * 4 * (N + 1)

    # Lower bound for control
    F_max = config['multiple_shooting']['F_max']
    u_lwb = [-F_max] * N
    u_uwb = [ F_max] * N

    # --- Constraints ---
    # constraints are represented by:
    # lbg <= g_list <= ubg
    # g_list represents constraint vector
    g_list = [] 
    lbg    = []
    ubg    = []

    # 1. Continuity (defect) constraints: s_{k+1} = F_integrate(s_k, u_k, T/N, nSubStep)
    for k in range(N):
        s_k1_pred = F_integrate(s_nodes[k], u_nodes[k]) # 
        defect    = s_nodes[k + 1] - s_k1_pred # defect must be 0
        g_list.append(defect)
        lbg += [0.0] * 4 # assign to 0 as it is equality constraints
        ubg += [0.0] * 4 # assign to 0 as it is equality constraints
    
    # 2. Initial conditions: position at [0,pi,0,0]
    g_list.append(s_nodes[0][0])   # x0 = 0
    g_list.append(s_nodes[0][1] - np.pi)   # theta0 = pi
    g_list.append(s_nodes[0][2])   # dx0 = 0
    g_list.append(s_nodes[0][3])   # dtheta0 = 0
    lbg += [0.0, 0.0, 0.0, 0.0] # assign to 0 as it is equality constraints
    ubg += [0.0, 0.0, 0.0, 0.0] # assign to 0 as it is equality constraints

    # 3. Terminal conditions: position at [0,0,0,0]
    g_list.append(s_nodes[-1][0])   # xN = 0
    g_list.append(s_nodes[-1][1])   # thetaN = 0
    g_list.append(s_nodes[-1][2])   # dxN = 0
    g_list.append(s_nodes[-1][3])   # dthetaN = 0
    lbg += [0.0, 0.0, 0.0, 0.0] # assign to 0 as it is equality constraints
    ubg += [0.0, 0.0, 0.0, 0.0] # assign to 0 as it is equality constraints

    # --- Objective ---
    # Minimize sum of squared control inputs
    dt = T / N # interval duration
    J = 0
    for k in range(N):
        J += u_nodes[k]**2 * dt

    # --- NLP problem setup ---
    nlp_prob = {'f': J, 'x': w, 'g': vertcat(*g_list)}
    opts = {'ipopt.print_level': 5, 'print_time': 1}
    solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

    sol = solver(x0=w0, lbx=s_lwb + u_lwb, ubx=s_uwb + u_uwb, lbg=lbg, ubg=ubg)

    # --- Extract solution ---
    w_opt = sol['x'].full().flatten()
    s_opt = w_opt[:4*(N+1)].reshape((N+1, 4)) # shape (N+1, 4)
    u_opt = w_opt[4*(N+1):].reshape(N) # shape (N,)


    # simulate the optimal trajectory
    cart_pole.simulate_state_trajectory(np.linspace(0, T, N+1), s_opt, isRender=True) # looks ok

        # Simulate forward dynamics with the optimal control trajectory using RK4 integrator to verify the solution
    nSubStepDense = 100 # number of sub-steps for dense simulation
    t_simu, x_simu = simulate_optimal_control(cart_pole, s_opt[0], u_opt, T, n_substeps=nSubStepDense)

    #compute the maximum defect between the simulated trajectory and the optimal trajectory at the shooting nodes
    max_defect = 0.0
    for k in range(N):
        # find the index in the simulated trajectory that corresponds to the shooting node time
        t_node = k * dt
        idx = np.argmin(np.abs(t_simu - t_node))
        defect = np.linalg.norm(x_simu[idx] - s_opt[k])
        max_defect = max(max_defect, defect)

    print(f"Maximum defect between simulated and optimal trajectory: {max_defect}")

    # plot the optimal trajectory
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, T, N+1), s_opt[:, 0], label='x (cart position)')
    plt.title('Optimal State Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('x (cart position)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, T, N+1), s_opt[:, 1], label='theta (pole angle)', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('theta (pole angle)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, T, N), u_opt, label='u (control force)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('u (control force)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Second plot: compare the simulated trajectory with the optimal trajectory
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_simu, x_simu[:, 0], label='x (simulated)', linestyle='dashed')
    plt.plot(np.linspace(0, T, N+1), s_opt[:, 0], label='x (optimal)', linestyle='solid')
    plt.xlabel('Time (s)')
    plt.ylabel('x (cart position)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_simu, x_simu[:, 1], label='theta (simulated)', linestyle='dashed', color='orange')
    plt.plot(np.linspace(0, T, N+1), s_opt[:, 1], label='theta (optimal)', linestyle='solid', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('theta (pole angle)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()



    