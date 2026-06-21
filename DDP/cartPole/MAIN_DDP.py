from cartPoledynamics import*
import yaml

if __name__ == "__main__":
    
    # Load parameter from yaml file
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    Nt = config['DDP']['Nt']
    T = config['DDP']['T']
    h = T/Nt
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
    
    # trajectory variable
    xtraj = np.zeros([dim_x, Nt]) # state trajectory vector, shape (4,Nt)
    utraj = np.zeros([1, Nt]) # shape (1, Nt)
    print("init ok")
    
    # Initial Rollout
    
    for i in range(Nt-1):
        xtraj[:,i+1] = cartPole.DiscreteEulerDynamics(xtraj[:,i],utraj[:,i].item(), h)
        
    print("Init initial Rollout")
    