import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class cartPole:
    def __init__(self, param):
        self.g = param['gravity']
        self.M = param['mass_cart']
        self.m = param['mass_pole']
        self.l = param['length']
        self.b_p = param.get('damping_pole', 0.0)
        self.b_c = param.get('damping_cart', 0.0)

    def dynamics(self, state, u):
        x, theta, dx, dtheta = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        ddx = (u + self.m * sin_theta * (self.l * dtheta**2 + self.g * cos_theta) - self.b_c * dx) / (self.M + self.m * sin_theta**2)
        ddtheta = (-u * cos_theta - self.m * self.l * dtheta**2 * sin_theta * cos_theta - (self.M + self.m) * self.g * sin_theta - self.b_p * dtheta) / (self.l * (self.M + self.m * sin_theta**2))

        return np.array([dx, dtheta, ddx, ddtheta])
    
    def simulate(self, init_state, u_traj, t_traj, isRender=False):
        # make sure u_traj and t_traj have the same length
        assert len(u_traj) == len(t_traj), "Control trajectory and time trajectory must have the same length."
        def ode(t, state):
            u = np.interp(t, t_traj, u_traj)
            return self.dynamics(state, u)

        sol = solve_ivp(ode, (t_traj[0], t_traj[-1]), init_state, t_eval=t_traj)
        if isRender:
            cart_width = 0.1
            cart_height = 0.075
            dt_ms = (t_traj[1] - t_traj[0]) * 1000

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.4, 0.4)
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
        pole_y = [0, -pole_length * np.cos(theta)]
        plt.plot(pole_x, pole_y, color='red', linewidth=2)
        # draw a circle at the end of the pole
        plt.plot(pole_x, pole_y, color='red', linewidth=2)
        plt.plot(pole_x[1], pole_y[1], 'ro', markersize=10)
        plt.xlim(-1, 1)
        plt.ylim(-0.4, 0.4)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Cart-Pole System")
        plt.grid()
        plt.show()