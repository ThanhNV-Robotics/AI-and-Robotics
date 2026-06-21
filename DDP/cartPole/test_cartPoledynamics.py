import numpy as np
import casadi as ca
from cartPoledynamics import cartPole
import yaml

if __name__ == "__main__":
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cartpole_obj = cartPole(config['dynamic_parameters'])

    # CasADi symbolic state and control
    x_sym = ca.MX.sym('x', 4)
    u_sym = ca.MX.sym('u')

    x_test = np.array([0.0, np.pi / 3, 0.0, 0.0])
    u_test = 0.0
    h = 0.02

    # --- Test 1: symbolic dynamics ---
    print("=== Test CasADi symbolic dynamics ===")
    dx_sym = cartpole_obj.dynamics(x_sym, u_sym)
    f_dyn = ca.Function('f_dyn', [x_sym, u_sym], [dx_sym])
    dx_val = np.array(f_dyn(x_test, u_test)).flatten()
    print("  dx at test point:", dx_val)

    # --- Test 2: discrete Euler dynamics ---
    print("\n=== Test DiscreteEulerDynamics ===")
    x_next_sym = cartpole_obj.DiscreteEulerDynamics(x_sym, u_sym, h)
    f_disc = ca.Function('f_disc', [x_sym, u_sym], [x_next_sym])
    x_next_val = np.array(f_disc(x_test, u_test)).flatten()
    print("  x_next at test point:", x_next_val)

    # --- Test 3: Jacobian dFdx ---
    print("\n=== Test dFdx (state Jacobian) ===")
    J_sym = cartpole_obj.dFdx(x_sym, u_sym, h)
    f_jac = ca.Function('f_jac', [x_sym, u_sym], [J_sym])
    J_val = np.array(f_jac(x_test, u_test))
    print("  Jacobian (4x4):")
    print(J_val)

    # --- Test 4: forward simulation (numerical, with render) ---
    print("\n=== Test simulateForwardDynamics ===")
    init_state = np.array([0.0, np.pi / 3, 0.0, 0.0])
    t = np.linspace(0, 10, 500)
    u_traj = 0 * t
    t_sol, state = cartpole_obj.simulateForwardDynamics(init_state, u_traj, t, isRender=True)
    print("  Simulation complete. State shape:", state.shape)
