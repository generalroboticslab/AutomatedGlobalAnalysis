import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, seed_everything


class DelayKoopDataset(Dataset):
    def __init__(self, 
                    prediction_horizon: int=40,
                    data_path: str='my/path'):
        super().__init__()
        '''
        data_path: path to numpy array of data 
        prediction_horizon: number time steps that will be predicted into the future
        '''

        # Ensure inputs are numpy arrays for easy manipulation
        data = np.load(data_path)
        self.state_data = data[0]
        self.state_dot_data = data[1]
        self.time_data = data[2]

        self.prediction_horizon = prediction_horizon

        # Calculate dimensions
        self.n_states, self.n_samples, self.n_traj = self.state_data.shape

    def __getitem__(self, index):
        '''
        The data is stored in a 2 3D arrays, where the first 2 dimensions of each data[:,:,i] is a windowed trajectory shaped like (n_states*n_delays, traj_len-n_delays).
        The third dimension is the number of trajectories.
        This function returns an initial (delayed) state x_t0 which is a column of state_data, 
        the derivative of the initial (delayed) state fx_t0 which is a column state_dot_data, 
        and the future states x_tn until the prediction horizon.
        '''
        i = index // (self.n_samples - self.prediction_horizon)
        j = index % (self.n_samples - self.prediction_horizon)

        x_t0 = self.state_data[:, j, i]
        fx_t0 = self.state_dot_data[:, j, i]
        x_tn = self.state_data[:, j+1:j+self.prediction_horizon, i]
        x_tf = self.state_data[:, j+self.prediction_horizon, i]
        fx_tf = self.state_dot_data[:, j+self.prediction_horizon, i]
        t0 = self.time_data[:, j, i]
        tn = self.time_data[:, j+1:j+self.prediction_horizon, i]
        tf = self.time_data[:, j+self.prediction_horizon, i]

        
        return x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf
        

    def __len__(self):
        return (self.n_samples - self.prediction_horizon) * self.n_traj


class DelayKoopDataModule(LightningDataModule):
    def __init__(self, batch_size: int=64, 
                 prediction_horizon: int=40, 
                 data_path: str='my/path',
                 seed: int=42,
                 train_ratio: float=0.7):
        super().__init__()
        '''
        state_data_path: path to numpy array of state data (position and velocity)
        statedot_data_path: path to numpy array of state derivative data (velocity and acceleration)
        prediction_horizon: number time steps that will be predicted into the future
        batch_size: number of samples per batch
        '''
        self.prediction_horizon = prediction_horizon
        self.data_path = data_path
        self.batch_size = batch_size
        self.seed = seed
        seed_everything(self.seed)

        self.dataset_size = len(DelayKoopDataset(prediction_horizon=self.prediction_horizon, 
                                            data_path=self.data_path))
        
        self.train_ratio = train_ratio
        self.train_size = int(self.train_ratio * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size
        self.steps_per_epoch = (self.train_size // self.batch_size) + 1

    def setup(self, stage: 'str'):
        if stage == "fit":
            seed_everything(self.seed)
            self.data = DelayKoopDataset(prediction_horizon=self.prediction_horizon, 
                                         data_path=self.data_path)


            train_size = int(self.train_ratio * len(self.data))
            gen = torch.Generator().manual_seed(self.seed)
            self.koop_train, self.koop_val = random_split(self.data, [train_size, self.val_size], generator=gen)

    def train_dataloader(self):
        return DataLoader(
            self.koop_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  
            pin_memory=True  
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.koop_val,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True
        )
