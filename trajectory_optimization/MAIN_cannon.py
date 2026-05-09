# This is the main script for optimal trajectory planning of the cannon example


# state: 
#   x = horizontal position
#   y = vertical position
#   dx = horizontal velocity
#   dy = vertical velocity
#   ddx = horizontal acceleration
#   ddy = vertical acceleration

# parameters:
#   c = drag coefficient
#   T = final simulation time
# Boundary conditions:
#   x(0) = 0,
#   y(0) = 0,
#   x(T) = xf, # constraint to hit the target
#   y(T) = yf  # constraint to hit the target

# Objective:
#  J = dx0^2 + dy0^2

# Dynamics constraint:
# cannon dynamics:
#   ddx = -c*dx*sqrt(dx^2 + dy^2)
#   ddy = -c*dy*sqrt(dx^2 + dy^2) - g

###----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from cannonDynamics import dynamics

def main():
    initSpeed = 9.0
    initAngle = np.deg2rad(45)

    # set the target, assume the start at (0, 0)
    targetX = 10.0
    targetY = 0.0
