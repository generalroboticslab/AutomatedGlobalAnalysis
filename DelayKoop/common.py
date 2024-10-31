from collections import namedtuple
import os
import numpy as np
import pytorch_lightning as pl
from sklearn.feature_selection import mutual_info_regression
from scipy import stats


ae_arch_params = ['decoder_hid_dim', 'decoder_n_layers', 'encoder_hid_dim', 'encoder_n_layers', 'embedding_dim', 'batch_norm', 'res_net', 'dropout_rate']
ae_arch_defaults = [128, 3, 128, 3, 30, False, True, None]
ae_params = namedtuple("ae_params", field_names=ae_arch_params, defaults=ae_arch_defaults)
"""
This is a namedtuple containing the parameters for the autoencoder architecture

Args:

decoder_hid_dim: number of hidden units in the decoder (default: 128)
decoder_n_layers: number of hidden layers in the decoder (default: 3)
encoder_hid_dim: number of hidden units in the encoder (default: 128)
encoder_n_layers: number of hidden layers in the encoder (default: 3)
embedding_dim: dimension of the embedding (default: 30)
batch_norm: whether or not to use batch normalization (default: False)
res_net: whether or not to use residual connections (default: True)
dropout_rate: dropout rate (default: None)

"""

data_params = ['n_delays', 'n_states', 'delta_t', 'HodHux']
data_defaults = [15, 2, .02, False]
data_params = namedtuple("data_params", field_names=data_params, defaults=data_defaults)
"""
This is a namedtuple containing the parameters for the data

Args:

n_delays: number of delays in the system (default: 15)
n_states: number of states in the system (default: 2)
delta_t: time step (default: .02)
HodHux: whether or not the dataset is the Hodgkin-Huxley model (default: False)

"""

loss_params = ['chain_rule_loss_wt', 'pred_loss_wt', 'recon_loss_wt', 'state_pred_loss_wt', 'gamma', 'G_reg_wt', 'full_jacobian']
loss_defaults = [.0001, 1, 1, 1, .9, 1e-3, False]
loss_params = namedtuple("loss_params", field_names=loss_params, defaults=loss_defaults)
"""
This is a namedtuple containing the parameters for the loss function

Args:

chain_rule_loss_wt: weight for the chain rule loss term (default: .0001)
pred_loss_wt: weight for the prediction loss terma in latent space (default: 1)
recon_loss_wt: weight for the reconstruction loss term (default: 1)
state_pred_loss_wt: weight for the long term prediction loss term in state-space (default: 1)
gamma: discount factor for the long term prediction loss term (default: .9)
G_reg_wt: weight for the Koopman Generator spectral regularization term (default: 1e-3)
full_jacobian: whether or not to use the full Jacobian (default: False)

"""

training_params = ['learning_rate', 'steps_per_epoch', 'max_epochs', 'prediction_horizon_train', 'prediction_horizon_val', 'downsample_factor']
training_defaults = [.0001, 1000, 150, 40, 80, 1]
training_params = namedtuple("training_params", field_names=training_params, defaults=training_defaults)
"""
This is a namedtuple containing the parameters for the training procedure

Args:

learning_rate: learning rate (default: .0001)
steps_per_epoch: number of steps per epoch (default: 1000)
max_epochs: maximum number of epochs to train for (default: 150)
prediction_horizon_train: number of steps to predict into the future during training (default: 40)
prediction_horizon_val: number of steps to predict into the future during validation (default: 80)
downsample_factor: int that determines how much to downsample the predictions by (default: 1)

"""

def frange_cycle_linear(n_iters, pretrain_period, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    if start-stop == 0:
        return np.ones(n_iters + pretrain_period) * stop
    
    L = np.ones(n_iters + pretrain_period) * stop
    L[:pretrain_period] = 0  # Setting the pretraining period coefficients to 0
    

    if ratio == 0:
        return L
    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period + pretrain_period) < n_iters + pretrain_period):
            L[int(i + c * period + pretrain_period)] = v
            v += step
            i += 1
            
    return L 

def frange_cycle_sigmoid(n_iters, pretrain_period, start = 0.0, stop = 1.0,  n_cycle=4, ratio=0.5):
    if start-stop == 0:
        return np.ones(n_iters + pretrain_period) * stop
    
    start0 = start
    start = 0.0
    stop0 = stop
    stop = 1.0
    
    L = np.ones(n_iters+pretrain_period)*stop
    L[:pretrain_period] = 0 # Setting the pretraining period coefficients to 0

    if ratio == 0:
        return L
    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop:
            idx = int(i+c*period) + pretrain_period
            L[idx] = 1.0/(1.0 + np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L*(stop0-start0) + start0

def frange_cycle_sigmoid_gamma(n_iters, pretrain_period, start = 0.0, stop = 1.0,  n_cycle=4, ratio=0.5):
    if start-stop == 0:
        return np.ones(n_iters + pretrain_period) * stop
    
    L = np.ones(n_iters+pretrain_period)*stop
    L[:pretrain_period] = 0 # Setting the pretraining period coefficients to 0

    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop:
            idx = int(i+c*period) + pretrain_period
            L[idx] = 1.0/(1.0+ np.exp(- (v*12.-2.)))
            v += step
            i += 1
    return L  

def frange_cycle_sigmoid_gamma_new(n_iters, pretrain_period):
    start_y = .6
    start_x = np.log(start_y/(1-start_y))
    xs = np.linspace(start_x, start_x+11., n_iters+pretrain_period)
    ys = 1.0/(1.0+ np.exp(-xs))

    return ys


class CyclicalAnnealingCallback(pl.Callback):
    def __init__(self, 
                 max_epochs, 
                 steps_per_epoch, 
                 pretrain_epochs, 
                 start=(0., 0., 0., 0., 0.), 
                 stop=(1., 1., 1., 1., 1.), 
                 n_cycle=(4, 4, 4, 4, 4), 
                 ratio=(.5, .5, .5, .5, .5), 
                 annealing=True,
                 reg_annealing=False, 
                 type='linear'):
        """
        Implements cyclical annealing for the loss weights and discount factor

        Args:

        max_epochs: maximum number of epochs to train for
        steps_per_epoch: number of steps per epoch
        pretrain_epochs: number of epochs to pretrain for
        start: start values for the schedule, tuple of length 5 (chain rule, prediction, state prediction, gamma, G_reg)
        stop: stop values for the schedule, tuple of length 5
        n_cycle: number of cycles for each parameter, tuple of length 5
        ratio: ratio of the linear schedule to the constant schedule, tuple of length 5
        annealing: whether or not to anneal the loss weights and discount factor
        reg_annealing: whether or not to anneal the regularization weights
        type: type of annealing, either 'linear' or 'sigmoid'
        """


        self.annealing = annealing
        self.reg_annealing = reg_annealing
        n_cycle1, n_cycle2, n_cycle3, n_cycle4, n_cycle5  = n_cycle
        start1, start2, start3, start4, start5 = start
        stop1, stop2, stop3, stop4, stop5 = stop
        ratio1, ratio2, ratio3, ratio4, ratio5 = ratio
        n_iter = max_epochs * steps_per_epoch
        self.pretrain_period = pretrain_epochs * steps_per_epoch

        if type == 'linear':
            self.schedule_1 = frange_cycle_linear(n_iter, self.pretrain_period, start1, stop1, n_cycle1, ratio1)
            self.schedule_2 = frange_cycle_linear(n_iter, self.pretrain_period, start2, stop2, n_cycle2, ratio2)
            self.schedule_3 = frange_cycle_linear(n_iter, self.pretrain_period, start3, stop3, n_cycle3, ratio3)
            self.schedule_4 = frange_cycle_linear(n_iter, self.pretrain_period, start4, stop4, n_cycle4, ratio4)
            self.schedule_5 = frange_cycle_linear(n_iter, self.pretrain_period, start5, stop5, n_cycle5, ratio5)
        elif type == 'sigmoid':
            self.schedule_1 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start1, stop1, n_cycle1, ratio1)
            self.schedule_2 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start2, stop2, n_cycle2, ratio2)
            self.schedule_3 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start3, stop3, n_cycle3, ratio3)
            self.schedule_4 = frange_cycle_sigmoid_gamma(n_iter, self.pretrain_period, start4, stop4, n_cycle4, ratio4)
            self.schedule_5 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start5, stop5, n_cycle5, ratio5)
        else:
            raise ValueError("type must be either 'linear' or 'sigmoid'")

        self.init_chain_rule_loss_wt = None
        self.init_pred_loss_wt = None
        self.init_state_pred_loss_wt = None
        self.init_gamma = None
        self.G_reg_wt = None

    def on_train_start(self, trainer, pl_module):
        self.init_chain_rule_loss_wt = pl_module.chain_rule_loss_wt
        self.init_pred_loss_wt = pl_module.pred_loss_wt
        self.init_state_pred_loss_wt = pl_module.state_pred_loss_wt
        self.init_gamma = pl_module.gamma
        self.init_G_reg_wt = pl_module.G_reg_wt

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step < self.pretrain_period or self.annealing:
            pl_module.chain_rule_loss_wt = self.schedule_1[trainer.global_step]
            pl_module.pred_loss_wt = self.schedule_2[trainer.global_step]
            pl_module.state_pred_loss_wt = self.schedule_3[trainer.global_step]
            pl_module.gamma = self.schedule_4[trainer.global_step]

            if self.reg_annealing:
                pl_module.G_reg_wt = self.schedule_5[trainer.global_step]
            
            else:
                pl_module.G_reg_wt = self.init_G_reg_wt

        else:
            pl_module.lin_loss_wt = self.init_lin_loss_wt
            pl_module.pred_loss_wt = self.init_pred_loss_wt
            pl_module.state_pred_loss_wt = self.init_state_pred_loss_wt
            pl_module.gamma = self.init_gamma
            pl_module.G_reg_wt = self.init_G_reg_wt


class MetricTracker(pl.Callback):
    def __init__(self):
        self.metric = []
        self.epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch > 0: # exclude the first epoch because it is just a sanity check
            logs = trainer.logged_metrics
            self.metric.append(logs['val_ss_pred_epoch'].item())
        self.epoch += 1



def get_mi(data_path, n_states, n_delays, n_neighbors_max, n_samples, seeds=(1, 2, 3, 4, 5)):
    """
    This function computes the mutual information between the states and the delayed states for each combination of states and delays
    """

    assert n_samples == len(seeds), "Number of seeds must equal number of samples"
    data = np.load(data_path)

    data = np.reshape(data, (n_states*n_delays, -1))
    # extract rows i in i*n_delay for i in range(n_states)
    Y = data[::n_delays, :]
    # delete rows i in i*n_delay  for i in range(n_states)
    X = np.delete(data, np.arange(0, data.shape[0], n_delays), axis=0)

    mi_list = []
    
    for sample in range(n_samples):
        np.random.seed(seeds[sample])
        random_idx = np.random.randint(0, X.shape[1], 500)
        X_temp = X[:, random_idx]
        Y_temp = Y[:, random_idx]
        # create a dict to add to the list
        mi_dict = {}
        for i in range(n_states):
            mi_dict[f"State {i}"] = []
            print(f"Sample {sample}")
            print(f"State {i}")
            for k in range(1, n_neighbors_max+1):
                x_idx = i*(n_delays-1)
                mi = mutual_info_regression(X_temp[x_idx:,:].T, Y_temp[i, :], n_neighbors=k)
                mi = np.reshape(mi, (n_states-i, n_delays-1))
                mi_dict[f"State {i}"].append(mi)
        mi_list.append(mi_dict)

        mi_avg, mi_se = avg_mi(mi_list, n_states, n_neighbors_max)

    return mi_avg, mi_se

def avg_mi(mi_list, n_states, n_neighbors_max):
    # iterate through each sample and each state and get the average MI for each delay and confidence interval
    mi_avg = []
    mi_se = []
    for i in range(n_states):
        mi_avg.append([])
        mi_se.append([])
        mi_state_i = []
        for j in range(n_neighbors_max-1):
            # concatenate all of the mi arrays for each sample and each nearest neighbor
            mis = np.stack([mi_list[s][f"State {i}"][j] for s in range(len(mi_list))], axis=0)
            if len(mi_state_i) == 0:
                mi_state_i = mis
            else:
                mi_state_i = np.concatenate((mi_state_i, mis), axis=0)

         # get the average and standard error of the mi
        mi_avg[i] = np.mean(mi_state_i, axis=0)
        mi_se[i] = stats.sem(mi_state_i, axis=0)

    mi_avg = np.concatenate(mi_avg, axis=0)
    mi_se = np.concatenate(mi_se, axis=0)

    return mi_avg, mi_se

def get_mi_random(data_path, n_states, n_delays, n_neighbors_max, n_samples, seeds=(1, 2, 3, 4, 5)):
    """
    This function computes the mutual information between the states and the delayed states for a random combination of states and delays
    when calculating the MI for every combination is too computationally expensive/intractable
    """
    assert n_samples == len(seeds), "Number of seeds must equal number of samples"
    data = np.load(data_path)

    data = np.reshape(data, (n_states*n_delays, -1))
    # extract rows i in i*n_delay for i in range(n_states)
    Y = data[::n_delays, :]
    # delete rows i in i*n_delay  for i in range(n_states)
    X = np.delete(data, np.arange(0, data.shape[0], n_delays), axis=0)

    mi_list = []    

    random_states = np.random.choice(n_states, 2, replace=False)
    middle_states = [n_states//2, n_states//2 + 1, n_states//2 - 1, n_states//2 - 2]
    sampled_states = middle_states #random_states.tolist() + middle_states

    n_states = len(sampled_states)

    lb = [i*(n_delays-1) for i in sampled_states]
    ub = [(i+1)*(n_delays-1) for i in sampled_states]

    results = []
    for lower, upper in zip(lb, ub):
        results.append(X[lower:upper, :])

    X = np.concatenate(results, axis=0)

    Y = Y[sampled_states, :]

    for sample in range(n_samples):
        np.random.seed(seeds[sample])
        random_idx = np.random.randint(0, X.shape[1], 500)
        X_temp = X[:, random_idx]
        Y_temp = Y[:, random_idx]
        # create a dict to add to the list
        mi_dict = {}
        for i in range(n_states):
            mi_dict[f"State {i}"] = []
            print(f"Sample {sample}")
            print(f"State {i}")
            for k in range(1, n_neighbors_max+1):
                x_idx = i*(n_delays-1)
                mi = mutual_info_regression(X_temp[x_idx:,:].T, Y_temp[i, :], n_neighbors=k)
                mi = np.reshape(mi, (n_states-i, n_delays-1))
                mi_dict[f"State {i}"].append(mi)
        mi_list.append(mi_dict)

        mi_avg, mi_se = avg_mi(mi_list, n_states, n_neighbors_max)

    return mi_avg, mi_se

def create_log_dir_structure(dataset_name, experiment_name, seed):
    """
    Creates the directory structure for a given experiment

    Args:

    dataset_name: name of the dataset
    experiment_name: name of the experiment
    seed: random seed
    
    """

    root_dir = "logs"
    experiment_dir = os.path.join(root_dir, f"{dataset_name}_{experiment_name}_{seed}")
    
    # Create main directory
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Create subdirectories
    subdirs = ["tb_logs", "checkpoints", "optuna", "csv"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir