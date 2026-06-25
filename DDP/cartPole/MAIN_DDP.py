from cartPoledynamics import*
import yaml
from DDP import DDP
if __name__ == "__main__":
    
    # Load parameter from yaml file
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    
    T = config['DDP']['T']
    h = config['dynamic_parameters']['h']

    Nt = np.uint16(T/h) + 1

    print("DDP configuration: ")
    print("Number of discreted point: ", Nt)
    print("Time horizon: ", T)
    print ("Time step: ", h)
    
    # create an instant of cartpole object
    cartPole = cartPole(config['dynamic_parameters'])
    
    # Initial condition / starting state
    # x, theta, xd, thetad
    x0 = np.array([0,0,0,0]) 
    dim_x = x0.shape[0]
    # target state
    xgoal = np.array([0,np.pi,0,0])

    # Cost weights
    Qk = config['DDP']['Qk'] * np.eye(dim_x)   # stage state cost
    QN = config['DDP']['QN'] * np.eye(dim_x)   # terminal state cost
    R  = config['DDP']['R']  * np.eye(1)        # control cost

    # Create a DDP object
    ddp_obj = DDP(
        config        = config['DDP'],
        Nx            = dim_x,
        Nu            = 1,
        dynamics      = cartPole.DiscreteEulerDynamics
    )
    print("Init DDP object done")

    # trajectory variable
    xtraj = np.zeros([dim_x, Nt]) # state trajectory vector, shape (4,Nt)
    utraj = np.zeros([1, Nt]) # shape (1, Nt)
    print("init ok")
    
    # Initial Rollout
    
    for i in range(Nt-1):
        xtraj[:,i+1] = np.array(cartPole.DiscreteEulerDynamics(xtraj[:,i],utraj[:,i])).flatten()
        
    print("Init initial Rollout")
    