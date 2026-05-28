import numpy as np
from casadi import *
import trajectory_optimization.cannon_shooting.dataClass as dataClass
from trajectory_optimization.cannon_shooting.cannonDynamics import simulateCannon
import matplotlib.pyplot as plt

class IterationCallback(Callback):
    """CasADi callback that records decision variables at every IPOPT iteration."""
    def __init__(self, name, nx, ng):
        Callback.__init__(self)
        self.nx = nx
        self.ng = ng
        self.iterates = []
        self.construct(name, {})

    def get_n_in(self):  return nlpsol_n_out()
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        n = nlpsol_out(i)
        if n == 'f':              return Sparsity.scalar()
        elif n in ('x', 'lam_x'): return Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'): return Sparsity.dense(self.ng)
        else:                      return Sparsity(0, 0)

    def eval(self, arg):
        x_idx = next(i for i in range(nlpsol_n_out()) if nlpsol_out(i) == 'x')
        self.iterates.append(np.array(arg[x_idx]).flatten().copy())
        return [0]


def cannonSingleShooting (guess: dataClass.GuessSingleShoot, target: dataClass.Target, param: dataClass.singleShootingParam):
    # INPUT:
    #  guess: dataClass.GuessSingleShoot()
    #  target: dataClass.Target()
    #  param: dataClass.param()

    # OUTPUT:
    #  soln.t
    #  soln.x
    #  soln.y
    #  soln.dx
    #  soln.dy

    # Run a simulation to get initial guess
    

    init = dataClass.Init(speed=guess.init_speed, angle=guess.init_angle)
    

    c = param.c # drag coefficient
    nGrid = param.nGrid

    trajectory = simulateCannon(init, param)

    # get an initial guess
    guess.dx0 = trajectory.dx[0] # initial velocity in x direction
    guess.dy0 = trajectory.dy[0] # initial velocity in y direction
    guess.T = trajectory.t[-1] # total time of flight

    # --- CasADi Nonlinear Programming ---

    # Symbolic cannon dynamics: d/dt [x, y, vx, vy]
    g_val = 9.81
    s_sym  = MX.sym('s', 4) # state: [x, y, vx, vy]
    vx_s   = s_sym[2] # velocity in x direction
    vy_s   = s_sym[3] # velocity in y direction
    v_s    = sqrt(vx_s**2 + vy_s**2) # speed (magnitude of velocity)

    ode_sym = vertcat(vx_s, vy_s,
                      -c * vx_s * v_s,
                      -c * vy_s * v_s - g_val) # cannon dynamics equation with drag

    # Time-scaled integrator: tau in [0, 1], d/dtau(z) = T * f(z)
    # Each of the nGrid steps covers 1/nGrid of the normalised interval
    T_p = MX.sym('T_p') # total time of flight (decision variable)
    dae  = {'x': s_sym, 'p': T_p, 'ode': T_p * ode_sym} # create an integrator in CasADi
    F    = integrator('F', 'cvodes', dae, {'tf': 1.0 / nGrid})

    # Decision variables: w = [dx0, dy0, T]
    dx0_var = MX.sym('dx0')
    dy0_var = MX.sym('dy0')
    T_var   = MX.sym('T')
    w  = vertcat(dx0_var, dy0_var, T_var)
    w0 = [guess.dx0, guess.dy0, guess.T]

    # Single shooting: propagate initial state through nGrid steps
    s = vertcat(MX(0.0), MX(0.0), dx0_var, dy0_var)
    for _ in range(nGrid):
        s = F(x0=s, p=T_var)['xf']

    x_final = s[0]
    y_final = s[1]

    # NLP: minimise launch speed squared, subject to hitting the target
    nlp  = {'x': w,
            'f': dx0_var**2 + dy0_var**2,
            'g': vertcat(x_final - target.x, y_final - target.y)}
    iter_cb = IterationCallback('iter_cb', 3, 2)
    opts = {'ipopt.print_level': 5, 'print_time': 1, 'iteration_callback': iter_cb}
    solver = nlpsol('solver', 'ipopt', nlp, opts)

    lbw = [-inf, 0.0, 0.1]  # dy0 >= 0 (upward launch), T > 0
    ubw = [inf,  inf, inf]
    lbg = [0.0, 0.0]        # equality: residuals must be zero
    ubg = [0.0, 0.0]

    sol   = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    dx0_opt, dy0_opt, T_opt = float(w_opt[0]), float(w_opt[1]), float(w_opt[2])

    print(f"Optimised: dx0={dx0_opt:.3f} m/s, dy0={dy0_opt:.3f} m/s, T={T_opt:.3f} s")

    # Reconstruct the optimised trajectory using simulateCannon
    init_opt       = dataClass.Init(speed=float(np.sqrt(dx0_opt**2 + dy0_opt**2)),
                                   angle=float(np.arctan2(dy0_opt, dx0_opt)))
    trajectory_opt = simulateCannon(init_opt, param)

    # Pick 3 evenly-spaced intermediate iterates (excluding the final one)
    all_iters = iter_cb.iterates
    n_pre = len(all_iters) - 1  # exclude last (converged) iterate
    if n_pre >= 3:
        indices = np.round(np.linspace(0, n_pre - 1, 3)).astype(int)
    else:
        indices = list(range(n_pre))

    intermediate_trajectories = []
    for idx in indices:
        dx0_i, dy0_i, _ = all_iters[idx]
        init_i       = dataClass.Init(speed=float(np.sqrt(dx0_i**2 + dy0_i**2)),
                                     angle=float(np.arctan2(dy0_i, dx0_i)))
        intermediate_trajectories.append(simulateCannon(init_i, param))

    return trajectory_opt, intermediate_trajectories

# test the function
if __name__ == "__main__":
    guess = dataClass.GuessSingleShoot(init_speed=9.0, init_angle=np.deg2rad(45))
    target = dataClass.Target(x=12.0, y=0.0)
    diagnostic = dataClass.diagonostic(enable=True)
    param = dataClass.dynamicsSimulation(c=0.05, nGrid=20)

    trajectory_opt, iter_trajs = cannonSingleShooting(guess, target, param)

    # plot optimization progress
    colors = ['#aec6e8', '#6baed6', '#2171b5']  # light -> dark blue for intermediate
    plt.figure()
    for i, traj in enumerate(iter_trajs):
        plt.scatter(traj.x, traj.y, color=colors[i], label=f'Iteration {i+1}')
    plt.scatter(trajectory_opt.x, trajectory_opt.y, color='black', s=50,
                label='Optimal')
    plt.scatter(target.x, target.y, color='red', s=200, zorder=5, label='Target')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(0, 14)
    plt.ylim(0, 10)
    plt.legend()

    plt.title("Single Shooting Optimization Progress")
    plt.show()



