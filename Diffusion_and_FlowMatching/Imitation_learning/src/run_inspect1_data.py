# Van Thanh Nguyen
# This code is for inspecting the training data set

import data as data_process
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio

ENV_ID = "gym_pusht/PushT-v0"

def render_an_episode(
    episode_states: np.ndarray,
    output_path: str | Path = "episode.mp4",
    fps: int = 10,
) -> None:
    """Replay a recorded episode by injecting states and save to a video file.

    Args:
        episode_states: Array of shape (T, 5) — recorded states [agent_x, agent_y, block_x, block_y, block_angle].
        output_path: Path to save the .mp4 file.
        fps: Frames per second for the output video (dataset is recorded at 10 Hz).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    env.reset(options={"reset_to_state": episode_states[0].tolist()})

    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for state in episode_states:
            env.unwrapped._set_state(state.astype(np.float64))
            frame = env.render()
            writer.append_data(frame)

    env.close()
    print(f"Saved to {output_path}")

def plot_state_data (state: np.asarray):
    step = np.arange(len(states))

    # print(Tobj_pos.shape)

    plt.figure()
    plt.plot(step,states[:,0]) # agent position x
    plt.title("state 0")

    plt.figure()
    plt.plot(step,states[:,1]) # agent position y
    plt.title("state 1")

    plt.figure()
    plt.plot(step,states[:,2]) # block x
    plt.title("state 2")

    plt.figure()
    plt.plot(step,states[:,3]) 
    plt.title("state 3")

    plt.figure()
    plt.plot(step,states[:,4])
    plt.title("state 4")
    plt.show()

def plot_an_action_curve_episode (action: np.asarray, episode_ends: np.asarray, episode_number: int):
    num_eps = len(episode_ends)

    assert 0 <= episode_number < num_eps, f"episode_number {episode_number} out of range [0, {num_eps - 1}]"

    start = episode_ends[episode_number - 1] if episode_number > 0 else 0 # starting index
    end = episode_ends[episode_number] # last index 

    episode_actions = action[start:end]

    plt.figure()
    plt.plot(episode_actions[:, 0], episode_actions[:, 1], marker='o', markersize=2)
    plt.scatter(episode_actions[0, 0], episode_actions[0, 1], color='green', zorder=5, label='start')
    plt.scatter(episode_actions[-1, 0], episode_actions[-1, 1], color='red', zorder=5, label='end')
    plt.title(f"Action path - Episode {episode_number}")
    plt.xlabel("Action x")
    plt.ylabel("Action y")
    plt.legend()
    plt.axis('equal')
    plt.show()

def get_an_episode_states (states: np.asarray, episode_ends: np.asarray, episode_number: int):
    num_eps = len(episode_ends)
    assert 0 <= episode_number < num_eps, f"episode_number {episode_number} out of range [0, {num_eps - 1}]"
    start = episode_ends[episode_number - 1] if episode_number > 0 else 0 # starting index
    end = episode_ends[episode_number] # last index 

    episode_states = states[start:end]

    return episode_states



if __name__ == "__main__":

    # download and unzip the pusht data set from the project's website

    data_dir: Path = Path("data")
    data_zarr_path = data_process.download_pusht(data_dir)

    print(data_zarr_path)

    states, actions, episode_ends = data_process.load_pusht_zarr(data_zarr_path)
    print("state shape is ", states.shape, "state dim: ", states.shape[1]) # state dim = 5
    print("action shape is ", actions.shape, "action dim: ", actions.shape[1]) # action dim = 2
    print("episode_end shape: ", episode_ends.shape)
    print ("sume of episode_end: ", np.sum(episode_ends))

    #-----------------------------------
    ######### Run the test function here
    #-----------------------------------
    
    #----------------------------------

    # plot_state_data(states)

    #----------------------------------
    
    # plot_an_action_curve_episode(actions,episode_ends,6)

    #----------------------------------

    episode_states = get_an_episode_states(states,episode_ends, 8)
    render_an_episode(episode_states)
    



