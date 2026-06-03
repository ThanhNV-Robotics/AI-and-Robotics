import numpy as np
import cartPoledynamics
import yaml

def ComputeGuessSwingUp (config,
                   cartpole_obj: cartPoledynamics.cartPole):
    # INPUT:
    #  config: configuration object containing parameters for the trajectory optimization
    #  cartpole_obj: an instance of cartPoledynamics.cartPole

    #  assumed initial state
    #  x_0 = np.array([0.0, 0.0, 0.0, 0.0]) # x, theta, dx, dtheta
    #  assumed final state
    #  x_f = np.array([0.0, np.pi, 0.0, 0.0]) # x, theta, dx, dtheta

    #  OUTPUT:
    #  guess_traj

    N = config['multiple_shooting']['N']
    T = config['multiple_shooting']['T']


    m = cartpole_obj.m
    M = cartpole_obj.M
    g = cartpole_obj.g
    l = cartpole_obj.l

    t = np.linspace(0, T, N)

    # assume the cart follows a sinusoidal trajectory, conserving CoM
    # CoM computation
    CoM = m * l / ( m + M)
    x = CoM * np.sin(np.pi * t / T) # at t=0, x=0; at t=T/2, x=CoM, at t=T, x=0
    dx = CoM * (2*np.pi / T) * np.cos(2 * np.pi * t / T) # derivative of x
    ddx = -CoM * (2*np.pi / T)**2 * np.sin(2 * np.pi * t / T) # second derivative of x

    # assume the pole/pendulumn goes from bottom (pi) to top (0) in one simple motion
    theta = np.pi*(1 - t/T)
    dtheta = -np.pi*np.ones_like(t)/T
    ddtheta = np.zeros_like(t)

    # now compute the inverse dynamics got get u_guess
    x_guess = np.vstack((x, theta, dx, dtheta)).T
    xd_guess = np.vstack((dx, dtheta, ddx, ddtheta)).T

    # append final state to ensure it ends at the desired upright position
    x_guess = np.vstack((x_guess, np.array([0.0, np.pi, 0.0, 0.0])))
    xd_guess = np.vstack((xd_guess, np.array([0.0, 0.0, 0.0, 0.0])))
    

    u_guess = np.zeros_like(t)

    for i in range(N):
        u_guess[i] = cartpole_obj.ComputeInverseDynamics(x_guess[i], xd_guess[i])
    
    t = np.append(t, T)

    return x_guess, xd_guess, u_guess, t

def simulate_optimal_control(cart_pole,
                             x0,
                             u_opt,
                             T,
                             n_substeps=20):
    """
    Forward simulate cart-pole using optimized controls.

    Parameters
    ----------
    cart_pole : cartPole object
        Your dynamics class.

    x0 : ndarray (4,)
        Initial state.

    u_opt : ndarray (N,)
        Optimized control sequence.

    T : float
        Total horizon.

    n_substeps : int
        RK4 steps inside each shooting interval.

    Returns
    -------
    t_nodes : ndarray (N+1,)
        Time at shooting nodes.

    x_nodes : ndarray (N+1,4)
        State at shooting nodes.
    """

    N = len(u_opt)
    dt = T / N
    h = dt / n_substeps

    x = np.array(x0, dtype=float)

    x_nodes = [x.copy()]
    t_nodes = [0.0]

    def rk4_step(x, u, h):

        k1 = cart_pole.dynamics(x, u)

        k2 = cart_pole.dynamics(x + 0.5*h*k1, u)

        k3 = cart_pole.dynamics(x + 0.5*h*k2, u)

        k4 = cart_pole.dynamics(x + h*k3, u)

        return x + h/6.0*(k1 + 2*k2 + 2*k3 + k4)

    for k in range(N):

        u = u_opt[k]      # zero-order hold

        for _ in range(n_substeps):
            x = rk4_step(x, u, h)

        x_nodes.append(x.copy())
        t_nodes.append((k+1)*dt)

    return np.array(t_nodes), np.array(x_nodes)
# For testing the ComputeGuessSwingUp function

if __name__ == "__main__":
    # parse yaml config file
    
    with open('trajectory_optimization/cart_pole/env_config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cart_pole = cartPoledynamics.cartPole(config['dynamic_parameters'])

    init_state = np.array([0.0, 0.0, 0.0, 0.0])  # x, theta, dx, dtheta

    # Compute initial guess trajectory
    x_guess, xd_guess, u_guess, t = ComputeGuessSwingUp(config, cart_pole)

    # Uncomment the following line to animate the guess trajectory
    # cart_pole.simulate_state_trajectory(t, x_guess, isRender=True) #

    # plote