import os
import pandas as pd
import sqlite3
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class BaseExperiment:
    def __init__(self, params):
        self.params = params
        self.experiment_name = params['experiment_name']
        self.data_path = params['data_path']
        self.seed = params['seed']
        self.n_trials = params['n_trials']
        self.n_delays = params['n_delays']
        self.accelerator = params['accelerator']
        self.gpu_idx = params['gpu_idx']
        self.batch_size = params['batch_size']
        self.max_lr = params['max_lr']
        self.pretrain_epochs = params['pretrain_epochs']
        self.main_epochs = params['main_epochs']
        self.n_states = params['n_states']
        self.train_horizon = params['train_horizon']
        self.val_horizon = params['val_horizon']
        self.downsample_fctr = params['downsample_fctr']
        self.encoder_hidden_dim = params['encoder_hidden_dim']
        self.decoder_hidden_dim = params['decoder_hidden_dim']
        self.encoder_hidden_layers = params['encoder_hidden_layers']
        self.decoder_hidden_layers = params['decoder_hidden_layers']
        self.dropout = params['dropout']
        self.latent_dim = self.params['latent_dim']
        self.sampler = self.params['sampler']
        self.full_jacobian = self.params['full_jacobian']
        assert self.sampler in ['random', 'tpe']
        self.dataset_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.create_log_dir_structure(self.dataset_name, self.experiment_name, self.seed)
        self.optuna_dir = os.path.join(self.log_dir, "optuna")
        self.tb_dir = os.path.join(self.log_dir, "tb_logs")
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.csv_dir = os.path.join(self.log_dir, "csv")


    def setup(self):
        pass

    def run(self):
        pass
    
    def create_log_dir_structure(self, dataset_name, experiment_name, seed):
        """
        Creates the directory structure for a given experiment

        Args:

        dataset_name: name of the dataset
        experiment_name: name of the experiment
        seed: random seed
        
        """
        root_dir = "logs"
        self.log_dir = os.path.join(root_dir, f"{dataset_name}_{experiment_name}_{seed}")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        subdirs = ["tb_logs", "checkpoints", "optuna", "csv"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)

    def get_optuna_data(self, db_path):
        """
        This function will get the data from the optuna database and return a pandas dataframe

        Args:

        db_path: path to the optuna database
        """
        
        con = sqlite3.connect(db_path)
        cursor = con.cursor()

        query = "SELECT * FROM trial_params;"
        cursor.execute(query)

        rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=['param_id', 'trial_id', 'param_name', 'param_value', 'distribution_json'])

        #print(df)

        pivot_df = df.pivot(index='trial_id', columns='param_name', values='param_value')

        #pivot_df.to_csv(f'{base_dir}/optuna_params.csv')

        print('Loading optuna data...')

        con.close()

        return pivot_df
    
    def get_tb_data(self, tb_log_dir, fields, versions=79):
        """
        This function will get the data from the tensorboard logs and return a pandas dataframe

        Args:

        tb_log_dir: base directory where the tensorboard logs are stored
        fields: list of fields to extract from the logs
        """
        df = pd.DataFrame(columns=fields)
        for version in range(versions+1):
            log_directory = tb_log_dir + f'/version_{version}'
            print(f'Loading data from version {version}...')
            event_acc = EventAccumulator(log_directory)
            event_acc.Reload()
            temp_df = pd.DataFrame(columns=fields)
            for field in fields:
                scalar = event_acc.Scalars(field)
                # get the last value and store in a dataframe
                temp_df[field] = [scalar[-1].value]

            df = pd.concat([df, temp_df], ignore_index=True)

        return df
    
    def teardown(self):
        db_path = f"{os.path.abspath(self.optuna_dir)}/{self.experiment_name}.db"
        
        optuna_df = self.get_optuna_data(db_path)
        optuna_df.index = optuna_df.index - 1
        fields = ['val_pred_epoch', 'val_recon_epoch', 'val_ss_pred_epoch', 'val_cr_epoch', 'n_pos_eigs_epoch', 'eig_clamp_epoch']
        tb_df = self.get_tb_data(os.path.abspath(self.tb_dir), fields, versions=self.n_trials-1)

        combined_df = pd.concat([tb_df, optuna_df], axis=1)

        #combined_df.drop(columns=['Unnamed: 0'], inplace=True)

        csv_path = os.path.join(self.csv_dir, f"main_results.csv")
        combined_df.to_csv(csv_path)


class ExperimentRunner:
    def __init__(self):
        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def run_all(self):
        for experiment in self.experiments:
            experiment.setup()
            print(f"Running {experiment.experiment_name}")
            experiment.run()
            print(f"Finished {experiment.experiment_name}")
            experiment.teardown()

