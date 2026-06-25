import numpy as np
import casadi as ca # to use auto differentiate fcn
from collections.abc import Callable

class DDP:
    def __init__(self, config, Nx, Nu, dynamics: Callable):

        # config: parse from the yaml file
        # Nx: state dimension
        # Nu: input dimension
        # dynamics: descreted dynamic function
        # terminal_cost: terminal cost function
        # stage_cost: stage cost function (lagrage term)

        self.T = config['T'] # terminal time
        self.Qk = config['Qk'] # termianl cost weight
        self.QN = config['QN'] # terminal cost weight
        self.Nx = Nx
        self.Nu = Nu

        # Cassadi symbolics variable
        x_sym = ca.MX.sym('x', Nx) # state variable, sym type
        u_sym = ca.MX.sym('u', Nu) # control variable, sym type

        # discrete dynamics — call the Python callable to get the symbolic expression
        Fd_sym = dynamics(x_sym, u_sym)
        self.Fd = ca.Function('fd', [x_sym, u_sym], [Fd_sym])

        # Jacobian and Hessian, compile once

        Jx_sym = ca.jacobian(Fd_sym, x_sym)   # (4x4) MX expression
        Ju_sym = ca.jacobian(Fd_sym, u_sym)   # (4x1) MX expression

        self.dFd_dx    = ca.Function('dFdx',    [x_sym, u_sym], [Jx_sym])
        self.dFd_du    = ca.Function('dFdu',    [x_sym, u_sym], [Ju_sym])

        # second derivatives: differentiate the symbolic Jacobian (not the compiled Function)
        self.d2Fd_dxdx = ca.Function('d2Fd_dxdx', [x_sym, u_sym], [ca.jacobian(Jx_sym, x_sym)]) 
        self.d2Fd_dxdu = ca.Function('d2Fd_dxdu', [x_sym, u_sym], [ca.jacobian(Jx_sym, u_sym)]) 
        self.d2Fd_dudu = ca.Function('d2Fd_dudu', [x_sym, u_sym], [ca.jacobian(Ju_sym, u_sym)])

    def stage_cost(self, x, u, xgoal):
        dx = x - xgoal
        return dx.T @ self.Qk @ dx + u.T @ self.R @ u

    def terminal_cost(self, x, xgoal):
        dx = x - xgoal
        return dx.T @ self.QN @ dx

    