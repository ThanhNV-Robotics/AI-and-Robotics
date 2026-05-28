from dataclasses import dataclass
import numpy as np
@dataclass
class GuessSingleShoot: # For single shooting
    init_speed: float
    init_angle: float
    dx0: float = 0.0 # initial velocity in x direction, to be filled by the simulation
    dy0: float = 0.0 # initial velocity in y direction, to be filled by the simulation
    T : float = 0.0 # total time of flight, to be filled by the simulation

@dataclass
class GuessMultipleShoot: # For multiple shooting
    T: float # flight duration
    t: np.ndarray # time grid
    z: np.ndarray # state grid, shape (4, nGrid), where z[0,:] is x, z[1,:] is y, z[2,:] is dx, z[3,:] is dy

@dataclass
class Init:
    speed: float
    angle: float

@dataclass
class Target:
    x: float
    y: float

@dataclass
class singleShooting:
    nGrid: int

@dataclass
class diagonostic:
    enable: bool

@dataclass
class dynamicsSimulation:
    c: float # drag coefficient
    nGrid: int 

@dataclass
class singleShootingParam:
    nGrid: int # number of grid points for the single shooting method
    c: float # drag coefficient
    diagnostic: diagonostic

@dataclass
class multipleShootingParam:
    nSegment: int
    nSubsStep: int

@dataclass
class traj:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray


@dataclass
class problem:
    x0: list # initial guess
    lb : list # lower bound on the decision variable
    ub : list # upper bound on the decision variable
    Aineq : list # inequality constraint matrix
    bineq : list # inequality constraint vector
    Aeq : list # equality constraint matrix
    beq : list # equality constraint vector