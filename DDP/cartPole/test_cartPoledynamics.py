import numpy as np
import casadi as ca
from cartPoledynamics import cartPole
import yaml

if __name__ == "__main__":
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cartpole_obj = cartPole(config['dynamic_parameters'])


    # --- Test 1: forward simulation (numerical, with render) ---
    print("\n=== Test simulateForwardDynamics ===")
    init_state = np.array([0.0, np.pi / 3, 0.0, 0.0])
    t = np.linspace(0, 5, 500)
    u_traj = 0 * t
    t_sol, state = cartpole_obj.simulateForwardDynamics(init_state, u_traj, t, isRender=True)
    print("  Simulation complete. State shape:", state.shape)

    # Test Jacobian matrix
    print("Testing Dynamic Jacobian Computation")

    # State boundary
    x_upper = np.array([2,2*np.pi,10,10])
    x_lower = np.array([-2,-2*np.pi,-10,-10])

    # Boundary for control input
    u_lower = -50
    u_upper = 50

    x_init = np.random.uniform(low=x_lower, high=x_upper)
    print("input state", x_init)

    u_init = np.random.uniform(u_lower,u_upper)
    print("control input: ", u_init)

    dFd_dx_cad = cartpole_obj.dFdx(x_init, u_init)
    print("Jacobian by cassadi:, ")
    print(dFd_dx_cad)

    print("Jacobian by finite different")

    dFd_dx_finite = cartpole_obj.finite_dFdx(x_init,u_init)
    print (dFd_dx_finite)

    # Hessian matrix
    print("Hessian d2F_dxdx by cassadi: ")
    d2F_dxdx = cartpole_obj.dF2_dxdx(x_init,u_init)
