import numpy as np
from casadi import *
import dataClass
from cannonDynamics import simulateCannon
import matplotlib.pyplot as plt

def cannonMultipleShooting(guess: dataClass.Guess, target: dataClass.Target, param: dataClass.multipleShootingParam):
    # Solving the cannon shooting problem using multiple shooting method
    # INPUT:
    #   guess.initSpeed
    #   guess.initAngle
    #   target.x
    #   target.y
    #   param.dynamics.c
    #   param.nGrid
    pass

if __name__ == "__main__":
    # Define the problem parameters
    target = dataClass.Target(x=100.0, y=0.0)
    init_guess = dataClass.Guess(init_speed=50.0, init_angle=np.pi/4)
    dynamics_param = dataClass.dynamics(c=0.1)
    single_shooting_param = dataClass.singleShooting(nGrid=20)
    diagnostic_param = dataClass.diagonostic(enable=True)
    param = dataClass.param(dynamics=dynamics_param, singleShooting=single_shooting_param, diagnostic=diagnostic_param)

    # Simulate the cannon trajectory with the initial guess
    traj = simulateCannon(init_guess, param)

    # Plot the trajectory
    plt.figure()
    plt.plot(traj.x, traj.y, label='Trajectory')
    plt.scatter(target.x, target.y, color='red', label='Target')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Cannon Trajectory')
    plt.legend()
    plt.grid()
    plt.show()