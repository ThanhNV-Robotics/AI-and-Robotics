import numpy as np
import cartPoledynamics



if __name__ == "__main__":
    # parse yaml config file
    import yaml
    with open('trajectory_optimization/cart_pole/env_config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cart_pole = cartPoledynamics.cartPole(config['dynamic_parameters'])
    # test_cart_pole_dynamics()
    init_state = np.array([0.0, np.pi/3, 0.0, 0.0])  # x, theta, dx, dtheta
    # cart_pole.draw_a_state(init_state)

    # Simulate for a short time with zero control input
    t_traj = np.linspace(0, 10, 500)  # simulate for 10 seconds
    # u_traj = 1 * np.sin(2*np.pi*0.5*t_traj)  # sinusoidal control input
    # u_traj = -1*t_traj  # linearly increasing control input

    u_traj = 0*t_traj  # zero control input

    t, states = cart_pole.simulateForwardDynamics(init_state, u_traj, t_traj, isRender=True)