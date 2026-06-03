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

    # # plot the guess trajectory
    # plt.figure(figsize=(12, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(t, x_guess[:, 0], label='x (cart position)')
    # plt.plot(t, x_guess[:, 1], label='theta (pole angle)')
    # plt.title('State Trajectory Guess')
    # plt.xlabel('Time (s)')
    # plt.ylabel('State')

    # plt.subplot(3, 1, 2)
    # plt.plot(t, x_guess[:, 2], label='dx (cart velocity)')
    # plt.plot(t, x_guess[:, 3], label='dtheta (pole angular velocity)')
    # plt.title('State Derivative Trajectory Guess')
    # plt.xlabel('Time (s)')
    # plt.ylabel('State Derivative')

    # plt.subplot(3, 1, 3)
    # plt.plot(t, u_guess, label='u (control input)')
    # plt.title('Control Trajectory Guess')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Control Input')
    # plt.legend()
    # plt.show()


    # simulate the guess trajectory
    # cart_pole.simulate_state_trajectory(t_guess, x_guess, isRender=True) # looks ok

    