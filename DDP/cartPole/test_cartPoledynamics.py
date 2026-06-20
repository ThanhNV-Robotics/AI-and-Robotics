import numpy as np
from cartPoledynamics import*
import yaml

if __name__ == "__main__":
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cartPole = cartPole(config['dynamic_parameters'])
    
    # test cartPole dynamics
    init_state = np.array([0.0, np.pi/3,0,0])
    
    # Simulate with 0 control input
    t = np.linspace(0,10,500)
    u_traj = 0*t
    
    t,state = cartPole.simulateForwardDynamics(init_state,u_traj,t,isRender=True)
        