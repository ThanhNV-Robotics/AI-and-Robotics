import numpy as np
import cartPoledynamics

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
    xAmp = m * l / ( m + M)
    x = xAmp * np.sin(np.pi * t / T) # at t=0, x=0; at t=T/2, x=xAmp, at t=T, x=0
    dx = xAmp * (2*np.pi / T) * np.cos(2 * np.pi * t / T) # derivative of x
    ddx = -xAmp * (2*np.pi / T)**2 * np.sin(2 * np.pi * t / T) # second derivative of x

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

# def 