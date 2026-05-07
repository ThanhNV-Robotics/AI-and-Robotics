import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

DRAG_COEFF = 0.01

class CanonShootingEnv:
    def __init__(self, drag_coeff):
        self.drag_coeff = drag_coeff
    
    
    def forward_dynamics (self,xd_0, yd_0, num_samples):
        def dynamic (t, state):
            # state s: x,y, xd, yd
            # dynamic: return d/dt (s) = f(s)
            g = 9.81 # m/s2, gravity const
            x = state[0]
            y = state[1]
            xd = state[2]
            yd = state[3]

            c = DRAG_COEFF


            s_dim = len(state)
            s_d = np.zeros(s_dim)
            s_d[0] = state[2] # d/dt (x)
            s_d[1] = state[3] # d/dt (y)
            
            v = np.sqrt(xd*xd + yd*yd)

            s_d[2] = -c*(xd*v)
            s_d[3] = -c*(yd*v) - g

            return s_d
            s_0 = np.array([0.0,0.0,xd_0,yd_0]) # initial condition

        x = None
        y = None
        time = None
        duration = 0

        initial_duration_guess = 5        
        
        s_0 = [0,0,xd_0, yd_0]

        while duration <= 0: # make sure that the simulation hit the ground
            t_duration = [0,initial_duration_guess]
            time_array = np.linspace(0, initial_duration_guess, num_samples)

            sol = solve_ivp(dynamic, t_span=t_duration, t_eval=time_array, y0 = s_0)
            
            y_terminal = sol.y[1,:][-1] # last element

            if y_terminal > 0: # not hit the ground yet
                initial_duration_guess +=1 # increase the simulation time
                continue
            else: # hit the ground, extract the duration and position data
                x = sol.y[0,:]
                y = sol.y[1,:]
                time = sol.t

                # speed before hitting the ground
                vx_before = sol.y[2,:][-1]
                vy_before = sol.y[3,:][-1]

                for i in range(len(y)):
                    if y[i] < 0:
                        time = time[:i]
                        x = x[:i]
                        y = y[:i]
                        duration = time[-1]
                        break
                        
                # estimate the final hitting point
                delta_t = np.abs((y[-1] - 0.0) /  vy_before)
                x_final =  x[-1] + 2*vx_before*delta_t
                y_final = 0.0
                t_final = time[-1] + delta_t
                
                x = np.append(x, x_final)
                y = np.append(y, y_final)
                time = np.append(time, t_final)
                duration += delta_t
                return time,x,y,duration
        return time,x,y,duration


def test_dynamics ():
    env = CanonShootingEnv(DRAG_COEFF)
    # initial speed:
    xd_0 = 15.0
    yd_0 = 2.0

    t,x,y,duration = env.forward_dynamics(xd_0,yd_0, 200)

    print("Simulation complete, duration = %.2f", duration)


    plt.figure()
    plt.scatter(x, y)

    plt.figure()
    plt.plot(t,x)
    plt.plot(t,y)

    plt.show()

if __name__ == "__main__":
    test_dynamics()
