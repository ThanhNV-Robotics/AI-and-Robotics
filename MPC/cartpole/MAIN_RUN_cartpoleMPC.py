from cartPoledynamics import *
import yaml
from MPC import MPC # import MPC class

if __name__ == "__main__":
    
    # Load parameters
    with open('MPC/cartpole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    cartpole_config = config['dynamic_parameters']
    mpc_config = config['MPC']
    # create a cartpole model
    
    carpole_model = cartPole(cartpole_config)
    print("Initiated a cartPole object")
    
    # Initial and target states: [x, theta, dx, dtheta]
    x0      = np.array([0.0, 0.0, 0.0, 0.0])
    xtarget = np.array([0.0, np.pi, 0.0, 0.0])
    Nx   = x0.shape[0]
    Nu = 1 # scalar
    
    # Initiate an MPC object
    MPC_sover = MPC(
        config = mpc_config,
        Nx = Nx,
        Nu = Nu,
        xgoal = xtarget,
        Ts = carpole_model.Ts,
        dynamics = carpole_model.dynamics
    )
    
    print("Initiated MPC")
    print("Continuous Linearized system matrix: Ac: \n", MPC_sover.Ac_)
    print("Continuous Linearized system matrix: Bc: \n", MPC_sover.Bc_)
    
    
    