import numpy as np
from casadi import *
import dataClass
from cannonDynamics import simulateCannon
import matplotlib.pyplot as plt

def cannonMultipleShooting(init: dataClass.Init, target: dataClass.Target, paramMultipleShooting: dataClass.multipleShootingParam, paramDynamics: dataClass.dynamicsSimulation):
    # Solving the cannon shooting problem using multiple shooting method

    # --- Initial guess via forward simulation ---
    guess_trajectory = simulateCannon(init, paramDynamics)

    n_segment  = paramMultipleShooting.nSegment
    n_sub_step = paramMultipleShooting.nSubsStep
    T_guess    = guess_trajectory.t[-1]

    t_collocation = np.linspace(0, T_guess, n_segment + 1)
    z_guess = np.zeros((4, n_segment + 1))
    z_guess[0, :] = np.interp(t_collocation, guess_trajectory.t, guess_trajectory.x)
    z_guess[1, :] = np.interp(t_collocation, guess_trajectory.t, guess_trajectory.y)
    z_guess[2, :] = np.interp(t_collocation, guess_trajectory.t, guess_trajectory.dx)
    z_guess[3, :] = np.interp(t_collocation, guess_trajectory.t, guess_trajectory.dy)

    initialGuess = dataClass.GuessMultipleShoot(T=T_guess, t=t_collocation, z=z_guess)

    # --- CasADi: ODE dynamics ---
    g_val = 9.81
    c     = paramDynamics.c

    s_sym = MX.sym('s', 4)   # state [x, y, vx, vy]
    h_sym = MX.sym('h')      # sub-step size

    vx_s    = s_sym[2]
    vy_s    = s_sym[3]
    v_s     = sqrt(vx_s**2 + vy_s**2)
    ode_rhs = vertcat(vx_s, vy_s, -c*vx_s*v_s, -c*vy_s*v_s - g_val)

    # Wrap ODE as a CasADi Function so it is auto-differentiable
    f_ode = Function('f_ode', [s_sym], [ode_rhs])

    # --- RK4 single-step Function ---
    k1 = f_ode(s_sym)
    k2 = f_ode(s_sym + 0.5*h_sym*k1)
    k3 = f_ode(s_sym + 0.5*h_sym*k2)
    k4 = f_ode(s_sym + h_sym*k3)
    s_next_rk4 = s_sym + (h_sym / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # F_step(s, h) -> s_next  (compiled CasADi Function, loop is unrolled at graph-build time)
    F_step = Function('F_step', [s_sym, h_sym], [s_next_rk4])

    def F_integrate(s0, T_seg, nSubStep):
        # Integrate over one shooting segment using nSubStep RK4 sub-steps
        h = T_seg / nSubStep
        s = s0
        for _ in range(nSubStep):
            s = F_step(s, h)
        return s

    # --- NLP decision variables: w = [s_0, s_1, ..., s_N, T] ---
    # Each s_k is a 4-vector; T is the total flight time (scalar)
    T_var   = MX.sym('T')
    s_nodes = [MX.sym(f's_{k}', 4) for k in range(n_segment + 1)]
    w       = vertcat(*s_nodes, T_var)

    # Initial guess: flatten state nodes (column-major) then append T
    w0  = np.concatenate([z_guess.T.flatten(), [T_guess]])
    lbw = [-np.inf] * len(w0)
    ubw = [ np.inf] * len(w0)
    lbw[-1] = 1e-2   # T > 0

    # --- Constraints ---
    # constraints are represented by:
    # lbg <= g_list <= ubg
    # g_list represents constraint vector
    g_list = [] 
    lbg    = []
    ubg    = []

    # 1. Continuity (defect) constraints: s_{k+1} = F_integrate(s_k, T/N)
    for k in range(n_segment):
        s_k1_pred = F_integrate(s_nodes[k], T_var / n_segment, n_sub_step)
        defect    = s_nodes[k + 1] - s_k1_pred # defect must be 0
        g_list.append(defect)
        lbg += [0.0] * 4 # assign to 0 as it is equality constraints
        ubg += [0.0] * 4 # assign to 0 as it is equality constraints

    # 2. Initial conditions: position at (0, 0)
    g_list.append(s_nodes[0][0])   # x0 = 0
    g_list.append(s_nodes[0][1])   # y0 = 0
    lbg += [0.0, 0.0] # assign to 0 as it is equality constraints
    ubg += [0.0, 0.0] # assign to 0 as it is equality constraints

    # 3. Terminal conditions: hit the target
    g_list.append(s_nodes[-1][0] - target.x)   # xN = target.x
    g_list.append(s_nodes[-1][1] - target.y)   # yN = target.y
    lbg += [0.0, 0.0] # assign to 0 as it is equality constraints
    ubg += [0.0, 0.0] # assign to 0 as it is equality constraints

    # --- Objective: minimize the required launch speed
    J = s_nodes[0][2]**2 + s_nodes[0][3]**2   # minimize |v0|^2 (equiv. to min |v0|, smoother)

    # --- Assemble and solve NLP ---
    nlp  = {'x': w, 'f': J, 'g': vertcat(*g_list)}
    opts = {'ipopt.print_level': 5, 'print_time': 1}
    solver = nlpsol('solver', 'ipopt', nlp, opts)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # --- Extract solution ---
    w_opt = sol['x'].full().flatten()
    s_opt = w_opt[:4 * (n_segment + 1)].reshape(n_segment + 1, 4).T  # shape (4, N+1)
    T_opt = float(w_opt[-1])
    t_opt = np.linspace(0, T_opt, n_segment + 1)

    return initialGuess, guess_trajectory, t_opt, s_opt, T_opt


if __name__ == "__main__":
    # Define the problem parameters
    target         = dataClass.Target(x=12.0, y=0.0)
    init_condition = dataClass.Init(speed=12.0, angle=np.pi/4)
    multiple_shooting_param = dataClass.multipleShootingParam(nSegment=10, nSubsStep=5)
    nsimuGrid = multiple_shooting_param.nSegment * multiple_shooting_param.nSubsStep
    dynamic_simulation_param = dataClass.dynamicsSimulation(c=0.05, nGrid=nsimuGrid)

    guess, guess_traj, t_opt, s_opt, T_opt = cannonMultipleShooting(
        init_condition, target, multiple_shooting_param, dynamic_simulation_param)

    vx0 = s_opt[2, 0]
    vy0 = s_opt[3, 0]
    v0  = np.sqrt(vx0**2 + vy0**2)
    print(f"Optimal flight time : {T_opt:.4f} s")
    print(f"Final position      : x = {s_opt[0, -1]:.4f} m,  y = {s_opt[1, -1]:.4f} m")
    print(f"Initial velocity    : vx0 = {vx0:.4f} m/s,  vy0 = {vy0:.4f} m/s")
    print(f"Initial speed |v0|  : {v0:.4f} m/s")

    plt.figure()
    plt.plot(guess_traj.x, guess_traj.y, 'b--', label='Initial guess')
    plt.plot(s_opt[0, :], s_opt[1, :], 'r-o', label='Optimized')
    plt.plot(target.x, target.y, 'k*', markersize=12, label='Target')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(True)
    plt.title('Multiple Shooting: Cannon Trajectory')
    plt.show()
    

    # Simulate the cannon trajectory with the initial guess and solve the NLP
    traj, t_collocation, z, T, traj_opt, z_opt, T_opt = cannonMultipleShooting(init_condition, target, multiple_shooting_param, dynamic_simulation_param)

    # Plot the trajectory
    plt.figure()
    plt.scatter(traj.x, traj.y, label='Initial Guess Trajectory') # simulated trajectory from the dynamics
    # plot the segment bound
    plt.scatter(z[0,:], z[1,:], color='green', label='Initial Segment Bound') # initial guess at the segment bound
    plt.scatter(z_opt[0,:], z_opt[1,:], color='orange', label='Optimised Segment Bound') # optimised segment bounds
    plt.plot(traj_opt.x, traj_opt.y, color='red', label='Optimised Trajectory')
    plt.scatter(target.x, target.y, color='red', marker='*', s=200, label='Target')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Cannon Trajectory')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()