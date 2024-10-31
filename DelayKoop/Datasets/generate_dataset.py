import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.interpolate as si
import numpy as np
import sys
import yaml
import os
import importlib


class GenerateDataset():
    def __init__(self, config_yaml_path):
        self.configs_yaml = config_yaml_path
        self.configs = None
        self.load_config()

        self.n_states = self.configs['n_states']
        self.delay_type = self.configs['delay_type']
        if (self.delay_type != 'dt') and (self.delay_type != 'dx'):
            raise Exception("Delay type not supported")
        self.delay_steps = self.configs['delay_steps']
        # if delay_type is dt then this is dt but if delay_type is dx then this is the arc length step size
        self.step_size = self.configs['step_size']
        self.noise_std = self.configs['noise_std']
        self.n_trajectories = self.configs['n_trajectories']
        # trajectory length in steps 
        self.trajectory_length = self.configs['trajectory_length']
        # duration of the simulation in seconds before interpolation
        self.sim_time = self.configs['sim_time']

        self.model_name = self.configs['model_name']
        # lower bounds and upper bounds for the initial conditions
        self.sample_bounds = self.configs['sample_bounds']

        # import model function from the model_name from the models file
        sys.path.append(os.getcwd())
        module = importlib.import_module(f"DelayKoop.Datasets.models")
        self.model = getattr(module, self.model_name)
        self.data_dir = self.configs['data_dir']
        self.save_dir = f"{self.data_dir}/{self.model_name}_delaytype-{self.delay_type}_delaysteps-{self.delay_steps}_stepsize-{self.step_size}_noise-{self.noise_std}_ntraj-{self.n_trajectories}_trajlen-{self.trajectory_length}"
        self.combined_dir = f"{self.data_dir}/{self.model_name}_delaytype-both_delaysteps-{self.delay_steps}_ntraj-{int(2*self.n_trajectories)}_trajlen-{self.trajectory_length}"
    def collect_data(self):
        windowed_s_matrices = []
        windowed_sdot_matrices = []
        windowed_t_matrices = []
        throw_away_count = 0
        traj_count = 0
        if self.model_name == 'linear_magnet_model':
            attractor_1_count = 0
            attractor_2_count = 0

        while traj_count < self.n_trajectories:
            if traj_count % 20 == 0:
                print(f"Trajectory count: {traj_count}")

            while True:
                throw_away = False
                t_, s_, s_dot_ = self.simulate_trajectory()
                throw_away, t, s, s_dot = self.interp_and_resample(t_, s_, s_dot_, traj_count)
                if not throw_away:
                    break
                throw_away_count += 1
                print(f"Trajectory thrown away")
            
            if self.model_name == 'linear_magnet_model':
                if traj_count == 0:
                    fixed_point_1 = s_[0, -1]
                    attractor_1_count += 1
                elif s_[0, -1] - fixed_point_1 < 0.1:
                    if attractor_1_count < self.n_trajectories/2:
                        attractor_1_count += 1
                    else:
                        continue
                else:
                    if attractor_2_count < self.n_trajectories/2:
                        attractor_2_count += 1
                    else:
                        continue

            windowed_s_matrix = np.empty((self.n_states*self.delay_steps, self.trajectory_length - self.delay_steps + 1))
            windowed_sdot_matrix = np.empty((self.n_states*self.delay_steps, self.trajectory_length - self.delay_steps + 1))
            windowed_t_matrix = np.empty((self.n_states*self.delay_steps, self.trajectory_length - self.delay_steps + 1))
            for i in range(self.n_states):
                # get the first state
                state = s[i, :]
                # get the first state derivative
                state_dot = s_dot[i, :]
                # get the windowed trajectory
                s_windowed = self.windowed_trajectory(state, self.delay_steps)
                s_dot_windowed = self.windowed_trajectory(state_dot, self.delay_steps)
                t_windowed = self.windowed_trajectory(t, self.delay_steps)
                # Append the windowed trajectory to the list
                windowed_s_matrix[i*self.delay_steps:(i+1)*self.delay_steps, :] = s_windowed
                windowed_sdot_matrix[i*self.delay_steps:(i+1)*self.delay_steps, :] = s_dot_windowed
                windowed_t_matrix[i*self.delay_steps:(i+1)*self.delay_steps, :] = t_windowed
                # Append the windowed matrix to the list
            
            windowed_s_matrices.append(windowed_s_matrix)
            windowed_sdot_matrices.append(windowed_sdot_matrix)
            windowed_t_matrices.append(windowed_t_matrix)
            traj_count += 1

        # save every 100 trajectories then concatenate them at the end
            if traj_count % 100 == 0:
                stacked_s_windowed_matrices = np.stack(windowed_s_matrices, axis=2)
                stacked_sdot_windowed_matrices = np.stack(windowed_sdot_matrices, axis=2)
                stacked_t_windowed_matrices = np.stack(windowed_t_matrices, axis=2)

                stacked_t_windowed_matrices = np.expand_dims(stacked_t_windowed_matrices, axis=0)

                combined_df = np.stack((stacked_s_windowed_matrices, stacked_sdot_windowed_matrices), axis=0)
                combined_df = np.concatenate((combined_df, stacked_t_windowed_matrices), axis=0)
                combined_df = combined_df.astype(np.float32)
                dir = f"{self.save_dir}_{traj_count}.npy"
                np.save(dir, combined_df)
                windowed_s_matrices = []
                windowed_sdot_matrices = []
                windowed_t_matrices = []

        # concatenate all the saved files
        combined_df = np.empty((3, self.n_states*self.delay_steps, self.trajectory_length - self.delay_steps + 1, self.n_trajectories))
        for i in range(0, self.n_trajectories+1, 100):
            if i ==0:
                continue
            data = np.load(f"{self.save_dir}_{i}.npy")
            combined_df[:, :, :, i-100:i] = data
            # delete the file
            import os
            os.remove(f"{self.save_dir}_{i}.npy")

        if len(windowed_s_matrices) > 0:
            stacked_s_windowed_matrices = np.stack(windowed_s_matrices, axis=2)
            stacked_sdot_windowed_matrices = np.stack(windowed_sdot_matrices, axis=2)
            stacked_t_windowed_matrices = np.stack(windowed_t_matrices, axis=2)

            stacked_t_windowed_matrices = np.expand_dims(stacked_t_windowed_matrices, axis=0)

            combined_df_end = np.stack((stacked_s_windowed_matrices, stacked_sdot_windowed_matrices), axis=0)
            combined_df_end = np.concatenate((combined_df_end, stacked_t_windowed_matrices), axis=0)
            combined_df_end = combined_df_end.astype(np.float32)
            # add the end data
            combined_df[:, :, :, i:] = combined_df_end

        combined_df = combined_df.astype(np.float32)
        if self.model_name == 'hodgkin_huxley':
            # dont overwrite the hodgkin huxley data if it already exists
            import os
            if not os.path.exists(f"{self.save_dir}.npy"):
                np.save(f"{self.save_dir}.npy", combined_df)
            else:
                # temporary fix to save the data cause the hodgkin huxley simulation tends to crash due to exploding gradients
                rand = np.random.randint(0, 1000)
                np.save(f"{self.save_dir}_{rand}.npy", combined_df)

        np.save(f"{self.save_dir}.npy", combined_df)
        print(f"Trajectories thrown away: {throw_away_count}")
        

    def load_data(self):
        return np.load(f"{self.save_dir}.npy")


    def simulate_trajectory(self):
        #orginal simulation dt
        dt = 0.005
        t_len = int(self.sim_time/dt)
        time = np.linspace(0, self.sim_time, t_len)
        if self.model_name == 'double_lco':
                r = np.random.uniform(0, 1.5)
                theta = np.random.uniform(0, 2*np.pi)
                x0 = [r*np.cos(theta), r*np.sin(theta)]
        elif self.model_name == 'van_der_pol':
            r = np.random.uniform(0, 5)
            theta = np.random.uniform(0, 2*np.pi)
            x0 = [r*np.cos(theta), r*np.sin(theta)]
        elif self.model_name == 'hodgkin_huxley':
            V0 = np.random.uniform(-100, 80)
            m0 = np.random.uniform(-2, 2)
            h0 = np.random.uniform(-2, 2)
            n0 = np.random.uniform(-2, 2)
            x0 = [V0, m0, h0, n0]
        elif self.model_name == 'lorenz_96':
            x0 = np.random.normal(0, 1, 40)

        else:
            # sample from beta distribution
            x0_unscaled = np.random.beta(.2, .2, self.n_states)  # #np.random.uniform(0, 1, self.n_states) #
            # scale the sample to the sample bounds
            x0 = [(self.sample_bounds[i+self.n_states]- self.sample_bounds[i])*x0_unscaled[i] + self.sample_bounds[i] for i in range(self.n_states)]

        # simulate dynamics
        sol = solve_ivp(self.model, t_span=[0, self.sim_time], y0=x0, args=(), t_eval=time)

        s = sol.y
        t = sol.t

        # get acceleration
        s_dot = self.get_s_dot(s)

        if self.noise_std > 0:
            s += np.random.normal(0, self.noise_std, s.shape)
            s_dot += np.random.normal(0, self.noise_std, s_dot.shape)

        return t, s, s_dot
    
    def get_s_dot(self, s):
        s_dot = np.empty(s.shape)
        for j in range(s.shape[1]):
                s_dot[:, j] = self.model(0, s[:, j])
        return s_dot
    
    def interp_and_resample(self, t, s, s_dot, sample):
        new_s = np.empty((self.n_states, self.trajectory_length))
        new_s_dot = np.empty((self.n_states, self.trajectory_length))
        t_new = np.empty(self.trajectory_length)
        throw_away = False
        # have to throw away the trajectory if the interpolation fails
        try:
            if self.delay_type == 'dt':
            # interpolate the trajectory as a function of time
            # resample the trajectory at the step size
                t_new = np.linspace(0, self.step_size*self.trajectory_length, self.trajectory_length)
                for i in range(self.n_states):
                    f = si.interp1d(t, s[i, :], 'cubic')
                    f_dot = si.interp1d(t, s_dot[i, :], 'cubic')
                    new_s[i, :] = f(t_new)
                    new_s_dot[i, :] = f_dot(t_new)    
            else:
            # interpolate the trajectory as a function of cumulative arc length
            # resample the trajectory at the arc length step size
                diff_s = np.diff(s, axis=1)
                arc_length = np.sqrt(np.sum(diff_s**2, axis=0))
                cum_arc_length = np.insert(np.cumsum(arc_length), 0, 0)
                starting_point = np.random.uniform(0, cum_arc_length[-1]-self.trajectory_length*self.step_size)
                arc_length_new = np.linspace(starting_point, starting_point+self.trajectory_length*self.step_size, self.trajectory_length)
                for i in range(self.n_states):
                    f = si.interp1d(cum_arc_length, s[i, :], 'cubic')
                    f_dot = si.interp1d(cum_arc_length, s_dot[i, :], 'cubic')
                    new_s[i, :] = f(arc_length_new)
                    new_s_dot[i, :] = f_dot(arc_length_new)
                
                t_new = si.interp1d(cum_arc_length, t, 'cubic')(arc_length_new)
        except:
            throw_away = True
        
        return  throw_away, t_new, new_s, new_s_dot
    
    def windowed_trajectory(self, trajectory, window_size):
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
    
    def load_config(self):
        try:
            with open(self.configs_yaml, 'r') as file:
                self.configs = yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load or parse the YAML file: {e}")
        



if __name__ == "__main__":
    config_yaml_path = './DelayKoop/Datasets/Configs/lorenz_96_configs.yaml'
    dataset = GenerateDataset(config_yaml_path)
    dataset.collect_data()
