import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca # to use auto differentiate fcn

class cartPole:
    def __init__(self, param):
        self.cart_width = 0.1
        self.cart_height = 0.075
        self.g = param['g']
        self.M = param['M']
        self.m = param['m']
        self.l = param['l']
        self.Ts = param['Ts']

        # Pre-build CasADi functions for Jacobians (compiled once, evaluated many times)
        x_sym = ca.MX.sym('x', 4) # state variable, sym type
        u_sym = ca.MX.sym('u') # control variable, sym type
        Fd_sym = self.DiscreteEulerDynamics(x_sym, u_sym)
        Jx_sym = ca.jacobian(Fd_sym, x_sym)   # (4x4) MX expression
        Ju_sym = ca.jacobian(Fd_sym, u_sym)   # (4x1) MX expression

        self.Fd = ca.Function('Fd',    [x_sym, u_sym], [Fd_sym]) # discreted dynamic eqn

        self._f_dFdx    = ca.Function('dFdx',    [x_sym, u_sym], [Jx_sym])
        self._f_dFdu    = ca.Function('dFdu',    [x_sym, u_sym], [Ju_sym])

        # second derivatives: differentiate the symbolic Jacobian (not the compiled Function)
        self._f_d2F_dxdx = ca.Function('d2F_dxdx', [x_sym, u_sym], [ca.jacobian(Jx_sym, x_sym)])  # (16x4)
        self._f_d2F_dxdu = ca.Function('d2F_dxdu', [x_sym, u_sym], [ca.jacobian(Jx_sym, u_sym)])  # (16x1)
        self._f_d2F_dudu = ca.Function('d2F_dudu', [x_sym, u_sym], [ca.jacobian(Ju_sym, u_sym)])  # (4x1)
    
    def dynamics(self, x, u):
        M = self.M
        m = self.m
        l = self.l
        g = self.g

        x1 = x[0] # x
        x2 = x[1] # theta
        x3 = x[2] # dx
        x4 = x[3] # dtheta

        s2 = np.sin(x2)
        c2 = np.cos(x2)

        dx1 = x3
        dx2 = x4

        dx3 = (u + m * l * x4**2 * s2  + m*g*c2*s2) / (M + m - m * c2**2)
        dx4 = -(g*(M+m)*s2 + c2*(u+m*l*x4**2*s2)) / (l*(M + m - m * c2**2))

        # we need to return cassadi variable to use auto differentiate
        return ca.vertcat(dx1, dx2, dx3, dx4)
    

    
    def DiscreteEulerDynamics(self, x, u):
        x_pred = x + self.Ts*self.dynamics(x,u)
        return x_pred

    def RolloutDiscreteDynamics (self,x0, utraj):
        # x0: initial state condition
        # utraj: a sequence of control input 
        # Compute sequence of states (forward dynamics)
        N = len(utraj)
        xtraj = np.zeros([4,N])
        xtraj[:,0] = x0 # append initial state to the trajectory

        for i in range(N+1): # compute from x1 to xN
            xtraj[:,i+1] = self.DiscreteEulerDynamics(xtraj[:,i], utraj[i])

        return xtraj
    
    def dFdx(self, x, u):
        return np.array(self._f_dFdx(x, u))
    
    # Second-order derivatives (for second-order DDP)
    # Returns reshaped (n_x, n_x, n_x) tensor — slice [:, :, i] = ∂²Fᵢ/∂x²
    def d2Fdxdx(self, x, u):
        return np.array(self._f_d2F_dxdx(x, u))   # (16x4)

    def d2Fdxdu(self, x, u):
        return np.array(self._f_d2F_dxdu(x, u))   # (16x1)

    def d2Fdudu(self, x, u):
        return np.array(self._f_d2F_dudu(x, u))   # (4x1)
    
    def finite_dFdx(self, x, u):
        eps = 1e-6
        n = len(x)
        dF_dx = np.zeros([n, n])
        F0 = np.array(self.DiscreteEulerDynamics(x, u)).flatten()

        for i in range(n):
            delta_x = np.zeros(n)
            delta_x[i] = eps                                          # perturb one element at a time
            Fi = np.array(self.DiscreteEulerDynamics(x + delta_x, u)).flatten()
            dF_dx[:, i] = (Fi - F0) / eps                            # i-th column = ∂F/∂xᵢ

        return dF_dx

    def dFdu(self, x, u):
        return np.array(self._f_dFdu(x, u))
    
    # def dF
    
    def ComputeInverseDynamics(self, state, dstate):
        # Compute control input u given state and desired acceleration
        # Can use it for a warm start
        M = self.M
        m = self.m
        l = self.l
        g = self.g

        x1 = state[0] # x
        x2 = state[1] # theta
        x3 = state[2] # dx
        x4 = state[3] # dtheta


        x1d = dstate[0] # dx
        x2d = dstate[1] # dtheta
        x3d = dstate[2] # ddx
        x4d = dstate[3] # ddtheta

        s2 = np.sin(x2)
        c2 = np.cos(x2)

        u = (M + m - m * c2**2) * x3d - m*l*x4**2*s2 - m*g*c2*s2
        return u
    
    def simulateForwardDynamics(self, init_state, u_traj, t_traj, isRender=True):
        # make sure u_traj and t_traj have the same length
        assert len(u_traj) == len(t_traj), "Control trajectory and time trajectory must have the same length."
        def ode(t, state):
            u = np.interp(t, t_traj, u_traj)
            return np.array(self.dynamics(state, u)).flatten()

        sol = solve_ivp(ode, (t_traj[0], t_traj[-1]), init_state, t_eval=t_traj)
        if isRender:
            cart_width = self.cart_width
            cart_height = self.cart_height
            dt_ms = (t_traj[1] - t_traj[0]) * 1000

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.6, 0.6)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Cart-Pole System")
            ax.grid()

            cart_patch = plt.Rectangle((0 - cart_width / 2, -cart_height / 2),
                                       cart_width, cart_height, color='blue')
            ax.add_patch(cart_patch)
            pole_line, = ax.plot([], [], color='red', linewidth=2)
            pole_bob, = ax.plot([], [], 'ro', markersize=10)

            def init():
                cart_patch.set_xy((-cart_width / 2, -cart_height / 2))
                pole_line.set_data([], [])
                pole_bob.set_data([], [])
                return cart_patch, pole_line, pole_bob

            def update(frame):
                x = sol.y[0, frame]
                theta = sol.y[1, frame]
                cart_patch.set_xy((x - cart_width / 2, -cart_height / 2))
                pole_x = [x, x + self.l * np.sin(theta)]
                pole_y = [0, -self.l * np.cos(theta)]
                pole_line.set_data(pole_x, pole_y)
                pole_bob.set_data([pole_x[1]], [pole_y[1]])
                return cart_patch, pole_line, pole_bob

            def on_finish(frame):
                update(frame)
                if frame == len(sol.t) - 1:
                    plt.close(fig)
                return cart_patch, pole_line, pole_bob

            _ = FuncAnimation(fig, on_finish, frames=len(sol.t),
                              init_func=init, interval=dt_ms, blit=True)
            plt.show()

        return sol.t, sol.y
    
    def simulate_state_trajectory(self, t_traj, x_traj, isRender=True):
        # make sure x_traj and t_traj have the same length
        # assert x_traj.shape[0] == len(t_traj), "State trajectory and time trajectory must have the same length."
        if isRender:
            cart_width = self.cart_width
            cart_height = self.cart_height
            dt_ms = (t_traj[1] - t_traj[0]) * 1000

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.6, 0.6)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Cart-Pole System")
            ax.grid()

            cart_patch = plt.Rectangle((0 - cart_width / 2, -cart_height / 2),
                                       cart_width, cart_height, color='blue')
            ax.add_patch(cart_patch)
            pole_line, = ax.plot([], [], color='red', linewidth=2)
            pole_bob, = ax.plot([], [], 'ro', markersize=10)

            def init():
                cart_patch.set_xy((-cart_width / 2, -cart_height / 2))
                pole_line.set_data([], [])
                pole_bob.set_data([], [])
                return cart_patch, pole_line, pole_bob

            def update(frame):
                x = x_traj[frame, 0]
                theta = x_traj[frame, 1]
                cart_patch.set_xy((x - cart_width / 2, -cart_height / 2))
                pole_x = [x, x + self.l * np.sin(theta)]
                pole_y = [0, -self.l * np.cos(theta)]
                pole_line.set_data(pole_x, pole_y)
                pole_bob.set_data([pole_x[1]], [pole_y[1]])
                return cart_patch, pole_line, pole_bob

            def on_finish(frame):
                update(frame)
                if frame == len(t_traj) - 1:
                    plt.close(fig)
                return cart_patch, pole_line, pole_bob

            _ = FuncAnimation(fig, on_finish, frames=len(t_traj),
                              init_func=init, interval=dt_ms, blit=True)
            plt.show()
        return

    
    def draw_a_state (self, state):
        # Draw the cart-pole system at a given state
        x, theta, _, _ = state
        cart_width = 0.1
        cart_height = 0.075
        pole_length = self.l
        plt.figure()
        # axis equal for proper aspect ratio
        plt.axis('equal')
        # Draw a rectangle for the cart
        plt.gca().add_patch(plt.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, fill=True, color='blue'))
        # Draw a line for the pole        
        pole_x = [x, x + pole_length * np.sin(theta)]
        pole_y = [0, pole_length * np.cos(theta)]
        plt.plot(pole_x, pole_y, color='red', linewidth=2)
        # draw a circle at the end of the pole
        plt.plot(pole_x, pole_y, color='red', linewidth=2)
        plt.plot(pole_x[1], pole_y[1], 'ro', markersize=10)
        plt.xlim(-1, 1)
        plt.ylim(-0.6, 0.6)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Cart-Pole System")
        plt.grid()
        plt.show()

