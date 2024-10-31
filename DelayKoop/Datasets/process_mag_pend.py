import sys
sys.path.insert(0, '/Users/Sam/Documents/DeepDelayKoopman')

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

def main():

    root_dir = './DelayKoop/Datasets/Experimental_Data/Single_Magnet_Pend'
    
    n_trajectories = 1000
    dt = 0.02
    n_states = 2
    delay = 31

    x_windowed_matrices = []
    xdot_windowed_matrices = []
    t_windowed_matrices = []

    for traj_idx in range(n_trajectories):
        pos = load_position(root_dir, traj_idx)

        if traj_idx == 0:
            t = np.linspace(0, 10.02, len(pos))

        pos, vel, acc, t_new = smooth_n_differentiate(pos, t, dt)

        x_flat = np.stack((pos, vel), axis=0)
        xdot_flat = np.stack((vel, acc), axis=0)
        t_new = np.expand_dims(t_new, axis=0)
        t_flat = np.concatenate((t_new, t_new), axis=0)


        window_n_stack(x_flat, xdot_flat, t_flat, delay, n_states, x_windowed_matrices, xdot_windowed_matrices, t_windowed_matrices)

    stacked_x_windowed_matrices = np.stack(x_windowed_matrices, axis=2)
    stacked_xdot_windowed_matrices = np.stack(xdot_windowed_matrices, axis=2)
    stacked_t_windowed_matrices = np.stack(t_windowed_matrices, axis=2)

    stacked_t_windowed_matrices = np.expand_dims(stacked_t_windowed_matrices, axis=0)

    stacked_x_windowed_matrices = stacked_x_windowed_matrices.astype(np.float32)
    stacked_xdot_windowed_matrices = stacked_xdot_windowed_matrices.astype(np.float32)

    combined_df = np.stack((stacked_x_windowed_matrices, stacked_xdot_windowed_matrices), axis=0)
    combined_df = np.concatenate((combined_df, stacked_t_windowed_matrices), axis=0)
    combined_df = combined_df.astype(np.float32)

    np.save(f'./Data/auto-mag-pend-{n_trajectories}traj-{dt}dt-{10}tf-{delay}delay', combined_df)


def smooth_n_differentiate(pos, t, dt):
        '''
        This function smooths the position data and then calculates the velocity and acceleration
        '''
        pos_smooth = sp.signal.savgol_filter(pos, 7, 3) # smooth the position data
        t_resample = np.arange(0.02, 10.02, dt)  # resample the time vector

        pos_interp = sp.interpolate.interp1d(t, pos_smooth, kind='cubic') # interpolate the position data
        pos_resample = pos_interp(t_resample) # resample the position data

        vel = (pos_smooth[1:] - pos_smooth[:-1]) / dt # calculate velocity

        t_shift = t + 0.01 # shift the time vector
        t_new = t_shift[0:-1] # remove the last element

        vel_interp = sp.interpolate.interp1d(t_new, vel, kind='cubic') # interpolate the velocity data
        vel_resample = vel_interp(t_resample) # resample the velocity data

        acc = (vel[1:] - vel[:-1]) / dt # calculate acceleration

        return pos_resample, vel_resample, acc, t_resample

def load_position(root_dir, traj_idx):
    """
    Loads a trajectory from the given root directory and trajectory index.
    
    Parameters:
    - root_dir: Root directory of the trajectory.
    - traj_idx: Index of the trajectory.
    
    Returns:
    - Numpy array of the trajectory.
    """
    t_x = np.loadtxt(f"{root_dir}/traj_{traj_idx}.csv", delimiter=",")

    return t_x[:, 1] # Return only the position values, not the time values or temperature values

def window_n_stack(x_flat, xdot_flat, t, delay, n_states, x_windowed_matrices, xdot_windowed_matrices, t_windowed_matrices):
    trajectory_length = x_flat.shape[1]

    x_windowed_matrix = np.empty((n_states*delay, trajectory_length - delay + 1))
    xdot_windowed_matrix = np.empty((n_states*delay, trajectory_length - delay + 1))
    t_windowed_matrix = np.empty((n_states*delay, trajectory_length - delay + 1))

    for i in range(n_states):
        # get the first state
        state = x_flat[i, :]
        # get the second state
        state_dot = xdot_flat[i, :]
        # get the time vector
        time = t[i, :]
        # get the windowed trajectory
        x_windowed = windowed_trajectory(state, delay)
        xdot_windowed = windowed_trajectory(state_dot, delay)
        # get the windowed time vector
        t_windowed = windowed_trajectory(time, delay)
        # Append the windowed trajectory to the list
        x_windowed_matrix[i*delay:(i+1)*delay, :] = x_windowed
        xdot_windowed_matrix[i*delay:(i+1)*delay, :] = xdot_windowed
        t_windowed_matrix[i*delay:(i+1)*delay, :] = t_windowed
        # Append the windowed matrix to the list

    return x_windowed_matrices.append(x_windowed_matrix), xdot_windowed_matrices.append(xdot_windowed_matrix), t_windowed_matrices.append(t_windowed_matrix)



def windowed_trajectory(trajectory, window_size):
    """
    Returns a matrix of windowed trajectories from the given trajectory using NumPy.
    
    Parameters:
    - trajectory: Array or list of values representing the trajectory.
    - window_size: Size of the window for the trajectory.
    
    Returns:
    - Numpy matrix of windowed trajectories.
    """
    if window_size <= 0 or window_size > len(trajectory):
        raise ValueError("Invalid window size.")

    trajectory = np.asarray(trajectory)  # Convert input to numpy array if it's not already
    n_windows = len(trajectory) - window_size + 1

    windowed = np.empty((window_size, n_windows))
    for i in range(n_windows):
        windowed[:, i] = trajectory[i:i+window_size][::-1]

    return windowed

if __name__ == '__main__':
    main()
