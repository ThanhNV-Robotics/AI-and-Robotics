import numpy as np
from casadi import *
import dataClass
from cannonDynamics import simulateCannon
import matplotlib.pyplot as plt

def cannonSingleShooting (guess: dataClass.Guess, target: dataClass.Target, param: dataClass.param):
    # INPUT:
    #  guess: dataClass.Guess()
    #  target: dataClass.Target()
    #  param: dataClass.param()

    # OUTPUT:
    #  soln.t
    #  soln.x
    #  soln.y
    #  soln.dx
    #  soln.dy

    # Run a simulation to get initial guess
    

    init = dataClass.Init()
    init.speed = guess.init_speed
    init.angle = guess.init_angle

    c = param.dynamics.c # drag coeff

    nGrid = param.singleShooting.nGrid

    trajectory = simulateCannon(init, param)

    # get an initial guess
    guess.dx0 = trajectory.dx[0] # initial velocity in x direction
    guess.dy0 = trajectory.dy[0] # initial velocity in y direction
    guess.T = trajectory.t[-1] # total time of flight
    return trajectory

# test the function
if __name__ == "__main__":
    guess = dataClass.Guess(init_speed=9.0, init_angle=np.deg2rad(45))
    target = dataClass.Target(x=10.0, y=0.0)
    dynamics = dataClass.dynamics(c=0.05)
    singleShooting = dataClass.singleShooting(nGrid=10)
    diagnostic = dataClass.diagonostic(enable=True)
    param = dataClass.param(dynamics=dynamics, singleShooting=singleShooting, diagnostic=diagnostic)

    trajectory = cannonSingleShooting(guess, target, param)

    # plot the trajectory
    x = trajectory.x
    y = trajectory.y
    plt.scatter(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(0, 14)
    plt.ylim(0, 10)
    # draw a red circle at the target
    plt.scatter(target.x, target.y, color='red',s=200, label='Target')
    plt.legend()
    plt.show()



