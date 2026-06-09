import numpy as np
from matplotlib import pyplot as plt

# randomize the action between -1 and 1 with gaussian noise
chunk_size = 100

action_rand = np.random.normal(loc=0.0, scale=1.0, size=(chunk_size, 2)) # A_t ~ N(0, 1)

tau = np.linspace(0, 1, chunk_size).reshape(-1, 1)

plt.scatter(action_rand[:, 0],action_rand[:, 1], label='Action 1')

plt.xlabel("Agent_x")
plt.ylabel("Agent_y")
plt.title("Random Action Chunk with Gaussian Noise")
plt.grid()
plt.show()