import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import trajectory_optimization.cannon_shooting.dataClass as dataClass

def dynamics (t, state, c):
    # state s: x,y, xd, yd
    # dynamic: return d/dt (s) = f(s)
    # c: drag coeff

    g = 9.81 # m/s2, gravity const
    x = state[0]
    y = state[1]
    xd = state[2]
    yd = state[3]

    s_dim = len(state)
    ds = np.zeros(s_dim)
    ds[0] = state[2] # d/dt (x)
    ds[1] = state[3] # d/dt (y)
    
    v = np.sqrt(xd*xd + yd*yd)

    ds[2] = -c*(xd*v)
    ds[3] = -c*(yd*v) - g

    return ds

def ground_event(t, z, c):
    # z[1] is the y-coordinate (altitude)
    # We return the value that we want to reach zero
    return z[1]

ground_event.terminal = True   # stop integration on event
ground_event.direction = -1    # only trigger when height is decreasing

def simulateCannon (init: dataClass.Init, param: dataClass.dynamicsSimulation):
    # INPUT:
    #  init: dataClass.Init()
    #  param: dataClass.dynamicsSimulation()

    # OUTPUT:
    #  soln.t
    #  soln.x
    #  soln.y
    #  soln.dx
    #  soln.dy

    v0 = init.speed
    th0 = init.angle

    c = param.c # drag coefficient
    nGrid = param.nGrid

    # set up initial condition
    x0 = 0.0
    y0 = 0.0

    dx0 = v0 * np.cos(th0)
    dy0 = v0 * np.sin(th0)

    # ensure dy0 > 0
    assert dy0 > 0, "Can not shoot through the ground, please set init_angle to be between 0 and 90 degree"

    duration = 5 # seconds
    tSpan = (0, duration)
    s0 = np.array([x0, y0, dx0, dy0])

    trajectory = dataClass.traj(
        t = np.zeros(nGrid),
        x = np.zeros(nGrid),
        y = np.zeros(nGrid),
        dx = np.zeros(nGrid),
        dy = np.zeros(nGrid)
    )

    # Run a simulation
    sol = solve_ivp(dynamics, tSpan, s0, events=ground_event, dense_output=True, args=(c,)) # dense_output=True to get a continuous solution

    # Time when the ball hits the ground (from event detection)
    t_impact = sol.t_events[0][0] if len(sol.t_events[0]) > 0 else sol.t[-1]

    # extract the solution on uniform grid
    t_grid = np.linspace(sol.t[0], t_impact, nGrid)
    z = sol.sol(t_grid)
    trajectory.t = t_grid
    trajectory.x = z[0, :]
    trajectory.y = z[1, :]
    trajectory.dx = z[2, :]
    trajectory.dy = z[3, :]

    return trajectory
    

def test_dynamics ():
    duration = 5 # seconds
    c = 0.05 # drag coeff
    xd_0 = 5.0
    yd_0 = 5.0

    t_span = (0, duration)
    s0 = np.array([0, 0, xd_0, yd_0])

    nGrid = 50

    # time = np.linspace(t_span[0], t_span[1], 100)

    sol = solve_ivp(dynamics, t_span, s0, dense_output=True, args=(c,)) # dense_output=True to get a continuous solution

    # extract the solution on uniform grid

    t_grid = np.linspace(sol.t[0], sol.t[-1], nGrid)
    z = sol.sol(t_grid)
    x = z[0, :]
    y = z[1, :]
    dx = z[2, :]
    dy = z[3, :]

    print("Simulation complete, duration = %.2f" % duration)

    target = (10, 0)


    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.xlim(0, 14)
    plt.ylim(0, 10)

    # draw a red circle at the target
    plt.scatter(target[0], target[1], color='red',s=200, label='Target')

    plt.show()

if __name__ == "__main__":
    test_dynamics()
