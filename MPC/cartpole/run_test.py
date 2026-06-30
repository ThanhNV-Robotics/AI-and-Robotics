import numpy as np
import casadi as ca
from cartPoledynamics import cartPole
import yaml

if __name__ == "__main__":
    with open('MPC/cartpole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cartpole_model = cartPole(config['dynamic_parameters'])

    # check linear dynamics
    xgoal = np.array([0,np.pi,0,0])
    Ac, Bc = cartpole_model.ComputeContinuousLinearizedMatrices(xgoal)
    print("Ac =\n", Ac)
    print("Bc =\n", Bc)