import numpy as np
import casadi as ca # to use auto differentiate fcn
from collections.abc import Callable
from time import time

class DDP:
    def __init__(self, config, Nx, Nu, Nt, xgoal, dynamics: Callable):

        # config: parse from the yaml file
        # Nx: state dimension
        # Nu: input dimension
        # Nt: number of collocation trajectory points
        # xgoal: final state goal
        # dynamics: descreted dynamic function
        # terminal_cost: terminal cost function
        # stage_cost: stage cost function (lagrage term)
        # this class solve for quadratics stage and terminal cost

        self.T        = config['T']        # terminal time
        self.Nx       = Nx                 # state dimension
        self.Nu       = Nu                 # control input dimension
        self.Nt       = Nt                 # number of collocation trajectory points
        self.max_iter = config['max_iter'] # max DDP iterations

        Qk = config['Qk']
        QN = config['QN']
        R = config['R']

        self.Qk =  Qk*np.diag(np.ones(Nx))# termianl cost weight, Nx x Nx, diagonal matrix
        self.QN = QN*np.diag(np.ones(Nx)) # terminal cost weight
        self.R = R*np.diag(np.ones(Nu)) # control cost weight

        self.xgoal = xgoal # desired terminal state

        # Cassadi symbolics variable
        x_sym = ca.MX.sym('x', Nx) # state variable, sym type
        u_sym = ca.MX.sym('u', Nu) # control variable, sym type

        # discrete dynamics — call the Python callable to get the symbolic expression
        Fd_sym = dynamics(x_sym, u_sym)
        self.Fd = ca.Function('Fd', [x_sym, u_sym], [Fd_sym])

        # Jacobian and Hessian, of the dynamics compile once

        Jx_sym = ca.jacobian(Fd_sym, x_sym)   # (4x4) MX expression
        Ju_sym = ca.jacobian(Fd_sym, u_sym)   # (4x1) MX expression

        self.dFd_dx    = ca.Function('dFdx',    [x_sym, u_sym], [Jx_sym])
        self.dFd_du    = ca.Function('dFdu',    [x_sym, u_sym], [Ju_sym])

        # second derivatives: differentiate the symbolic Jacobian (not the compiled Function)
        self.d2Fd_dxdx = ca.Function('d2Fd_dxdx', [x_sym, u_sym], [ca.jacobian(Jx_sym, x_sym)]) 
        self.d2Fd_dxdu = ca.Function('d2Fd_dxdu', [x_sym, u_sym], [ca.jacobian(Jx_sym, u_sym)]) 
        self.d2Fd_dudu = ca.Function('d2Fd_dudu', [x_sym, u_sym], [ca.jacobian(Ju_sym, u_sym)])

    def stage_cost(self, x, u):
        dx = x - self.xgoal
        return dx.T @ self.Qk @ dx + u.T @ self.R @ u

    def terminal_cost(self, x):
        dx = x - self.xgoal
        return dx.T @ self.QN @ dx
    
    def cost (self, xtraj, utraj):
        J = 0 # initialize for accumulated sum

        for i in range(self.Nt - 1): # compute the cost-to-go
            J += self.stage_cost(xtraj[:,i], utraj[:,i])

        J += self.terminal_cost(xtraj[:,-1]) # add the terminal cost

        return J
    
    def backward_pass(self, xtraj, utraj, mu=1e-3):
        Nx = self.Nx
        Nu = self.Nu
        Nt = self.Nt
        xgoal = self.xgoal

        Vx  = np.zeros((Nx, Nt))
        Vxx = np.zeros((Nx, Nx, Nt))
        d   = np.zeros((Nu, Nt - 1))
        K   = np.zeros((Nu, Nx, Nt - 1))

        # terminal boundary condition
        Vx[:, -1]    = self.QN @ (xtraj[:, -1] - xgoal)
        Vxx[:, :, -1] = self.QN

        delta_J = 0.0

        for k in range(Nt - 2, -1, -1):
            x = xtraj[:, k]
            u = utraj[:, k]

            dl_dx = self.Qk @ (x - xgoal)
            dl_du = self.R  @ u

            A = np.array(self.dFd_dx(x, u))   # (Nx, Nx)
            B = np.array(self.dFd_du(x, u))   # (Nx, Nu)

            Qx  = dl_dx + A.T @ Vx[:, k + 1]
            Qu  = dl_du + B.T @ Vx[:, k + 1]
            Qxx = self.Qk + A.T @ Vxx[:, :, k + 1] @ A
            Quu = self.R   + B.T @ Vxx[:, :, k + 1] @ B
            Qxu = A.T @ Vxx[:, :, k + 1] @ B
            Qux = B.T @ Vxx[:, :, k + 1] @ A

            # regularize Quu to keep it positive definite
            Quu_reg = Quu + mu * np.eye(Nu)

            dk = -np.linalg.solve(Quu_reg, Qu)
            Kk = -np.linalg.solve(Quu_reg, Qux)

            d[:, k]    = dk
            K[:, :, k] = Kk

            Vx[:, k]    = Qx + Kk.T @ Qu  + Qxu @ dk + Kk.T @ Quu @ dk
            Vxx[:, :, k] = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qxu @ Kk

            delta_J += dk.T @ Qu + 0.5 * dk.T @ Quu @ dk

        return Vx, Vxx, d, K, delta_J
    
    def forward_pass(self, X, U, d, K, alpha=1.0):
        Nt = self.Nt
        X_new = np.zeros_like(X)
        U_new = np.zeros_like(U)
        X_new[:, 0] = X[:, 0]  # same initial state

        for k in range(Nt - 1):
            delta_x = X_new[:, k] - X[:, k]
            U_new[:, k] = U[:, k] + alpha * d[:, k] + K[:, :, k] @ delta_x
            X_new[:, k+1] = np.array(self.Fd(X_new[:, k], U_new[:, k])).flatten()

        return X_new, U_new, self.cost(X_new, U_new)

    def optimize(self, x0, X_normal, U_normal):
        start_time = time()
        X = X_normal.copy()
        U = U_normal.copy()
        X[:, 0] = x0   # enforce fixed initial state
        last_cost = self.cost(X, U)
        print(f"DDP start: J={last_cost:.4f}")

        for i in range(self.max_iter):
            # Backward pass
            _, _, d, K, _ = self.backward_pass(X, U)

            # Forward pass with backtracking line search
            alpha = 1.0
            J_new = np.inf
            for _ in range(16):
                X_new, U_new, J_try = self.forward_pass(X, U, d, K, alpha)
                if np.isfinite(J_try) and J_try < last_cost:
                    J_new = J_try
                    break
                alpha *= 0.5

            if not np.isfinite(J_new):
                print(f"  iter {i+1:3d}: line search failed, skipping.")
                continue

            improvement = last_cost - J_new
            print(f"  iter {i+1:3d}: J={J_new:.4f}  dJ={improvement:.4e}  alpha={alpha:.4f}")

            X, U = X_new, U_new
            last_cost = J_new

            if abs(improvement) < 1e-4:
                print("Converged.")
                break

        elapsed = time() - start_time
        print(f"DDP done in {elapsed:.2f}s  final J={last_cost:.4f}")
        return X, U, last_cost





    