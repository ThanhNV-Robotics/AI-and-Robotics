import numpy as np
import casadi as ca
from collections.abc import Callable

class MPC:
    
    def __init__(self, config, Nx, Nu, xgoal, Ts, dynamics: Callable):
        self.P_ = config['P'] # terminal state weight
        self.Q_ = config['Q'] # stage state weight
        self.r_ = config['r'] # control input weight
        self.umax_ = config['umax'] # boudary on control input
        self.xmax_ = config['xmax'] # boundary for cart position
        
        self.Nx_ = Nx # state dimension
        self.Nu_ = Nu # control input dimension
        self.xgoal_ = xgoal # target terminal state
        
        self.Ts_ = Ts
        
        self.dynamics_ = dynamics # system dynamic, continuous form
        
        # compute A and B matrix for linearized dynamics
        
        self.Ac_, self.Bc_ = self.ComputeContinuousLinearizedMatrices()
        
    # Compute A and B matrix for continuous Linearized Dynamics
    def ComputeContinuousLinearizedMatrices(self):
        # xgoal: the target equilibrium point to linearize about
        # Returns Ac (4x4) and Bc (4x1) evaluated at (xgoal, u=0)
        xgoal = self.xgoal_

        # Pure symbols — CasADi jacobian() requires purely symbolic diff variables
        x_hat_sym = ca.MX.sym('x_hat', self.Nx_)
        u_sym     = ca.MX.sym('u', self.Nu_)

        # Substitute x = x_hat + xgoal into the dynamics
        f_sym = self.dynamics_(x_hat_sym + xgoal, u_sym)

        # Differentiate w.r.t. the pure symbols (not the derived expression x_hat+xgoal)
        Ac_sym = ca.jacobian(f_sym, x_hat_sym)
        Bc_sym = ca.jacobian(f_sym, u_sym)

        Ac_fcn = ca.Function('Ac', [x_hat_sym, u_sym], [Ac_sym])
        Bc_fcn = ca.Function('Bc', [x_hat_sym, u_sym], [Bc_sym])

        # Evaluate at equilibrium: deviation = 0, u = 0
        Ac = np.array(Ac_fcn(np.zeros(4), 0))
        Bc = np.array(Bc_fcn(np.zeros(4), 0))

        return Ac, Bc
        