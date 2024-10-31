from .base import BaseExperiment
from .. import common as cm
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ..pl_module import DelayKoop 
from ..pl_datamodule import DelayKoopDataModule
import pytorch_lightning as pl
import optuna
import os


 
class HyperOpt_StdAnneal(BaseExperiment):
    def __init__(self, params):
        
        """
        This experiment is designed to tune the loss hyperparameters (weights and discount factor) of a model
        that uses standard annealing for the loss weights and cyclical annealing for the discount factor. The model trained with annealing as opposed to 
        fixed weights has far less variance and the is less sensitive to the hyperparameters overall.

        The hyperparameters to tune are:
        - gamma n_cycles: how many times to cycle the discount factor for the state prediction loss
        - gamma ratio: what percentage of the total iterations should be spent with gamma=1.0
        - chain_rule_ratio: what percentage of the total iterations should be spent with the chain rule loss at 1.0
        - lat_pred_ratio: what percentage of the total iterations should be spent with the latent prediction loss at 1.0
        - state_pred_ratio: what percentage of the total iterations should be spent with the state prediction loss at 1.0

        Args:
            params (dict): Dictionary of parameters for the experiment.
            The parameters are:
            - experiment_name (str): Name of the experiment.
            - data_path (str): Path to the training state data.
            - seed (int): Random seed.
            - latent_dim (int): Dimension of the latent space.
            - n_delays (int): Number of delays.
            - n_trials (int): Number of trials to run.  
            - accelerator (str): Accelerator to use. 'gpu', 'tpu' or 'cpu'.
            - sampler (str): Which optuna sampler to use. 'random' or 'tpe'.
            - gpu_idx (int): Index of the GPU to use.
        """
        super().__init__(params)
        self.params = params

        self.objective = None
        self.total_epochs = self.pretrain_epochs + self.main_epochs
        self.lr = params['lr']
       
    def setup(self):
        """
        This method sets up the experiment by creating the data module, model, and objective function for optuna.
        """

        seed_everything(self.seed)
        dm = DelayKoopDataModule(data_path=self.data_path,
                                  batch_size=self.batch_size, 
                                  prediction_horizon=self.val_horizon, # for the validation forward pass
                                  seed=self.seed)
        
        ae_p = cm.ae_params(decoder_hid_dim=self.decoder_hidden_dim,
                         encoder_hid_dim=self.encoder_hidden_dim,
                         encoder_n_layers=self.encoder_hidden_layers,
                         decoder_n_layers=self.decoder_hidden_layers,
                         embedding_dim=self.latent_dim,
                         dropout_rate=self.dropout
        )

        data_p = cm.data_params(n_delays=self.n_delays,
                             n_states=self.n_states
        )

        training_p = cm.training_params(learning_rate=self.lr,
                                        steps_per_epoch=dm.steps_per_epoch,
                                        max_epochs = self.total_epochs,
                                        prediction_horizon_val=self.val_horizon, # for the validation forward pass
                                        prediction_horizon_train=self.train_horizon # for the training forward pass
        )

        loss_p = cm.loss_params(full_jacobian=self.params['full_jacobian'])

        def objective(trial):
            # to run a specific trial, the trial range is narrowed
            chain_rule_ratio = trial.suggest_float("chain_rule_ratio", 0.2, .8)
            lat_pred_ratio = trial.suggest_float("lat_pred_ratio", 0.2, .8)
            state_pred_ratio = trial.suggest_float("state_pred_ratio", 0.2, .8)
            gamma_ratio = trial.suggest_float("gamma_ratio", 0.5, .9)
            mu_ratio = trial.suggest_float("mu_ratio", 0.2, .8)

            print(f"chain_rule_ratio: {chain_rule_ratio}")
            print(f"lat_pred_ratio: {lat_pred_ratio}")
            print(f"state_pred_ratio: {state_pred_ratio}")
            print(f"gamm_ratio: {gamma_ratio}")
            print(f"mu_ratio: {mu_ratio}")

            cyclical_annealing_callback = cm.CyclicalAnnealingCallback(max_epochs=self.main_epochs, 
                                                                       steps_per_epoch=dm.steps_per_epoch,
                                                                       pretrain_epochs=self.pretrain_epochs,
                                                                       n_cycle=(1, 1, 1, 1, 1), 
                                                                       ratio=(chain_rule_ratio, lat_pred_ratio, state_pred_ratio, gamma_ratio, mu_ratio), 
                                                                       annealing=True,
                                                                        reg_annealing=True, 
                                                                        type='sigmoid')
            
            logger = TensorBoardLogger(self.log_dir, name='tb_logs')

            model = DelayKoop(ae_params=ae_p,
                            data_params=data_p,
                            loss_params=loss_p,
                            training_params=training_p)

            val_loss_tracker = cm.MetricTracker()

            ckpt_callback = ModelCheckpoint(dirpath=self.checkpoint_dir,
                                            filename=f"best_trial_{trial.number}")

            trainer = Trainer(logger=logger,
                              max_epochs=self.total_epochs,
                              accelerator=self.accelerator,
                              devices=[self.gpu_idx],
                                callbacks=[cyclical_annealing_callback, val_loss_tracker, ckpt_callback],
                                check_val_every_n_epoch=1)
            
            trainer.fit(model, dm)
            
            val_loss = val_loss_tracker.metric[-1]

            if trial.number > 0:
                best_val_loss = trial.study.best_trial.value

                best_trial = trial.study.best_trial.number

                if val_loss < best_val_loss:
                    os.remove(f"{self.checkpoint_dir}/best_trial_{best_trial}.ckpt")
                else:
                    os.remove(f"{self.checkpoint_dir}/best_trial_{trial.number}.ckpt")
            
            return val_loss

        self.objective = objective

    def run(self):
        """
        This method runs the experiment by creating a study and running the trials.
        """
        storage_name = storage_name = f"sqlite:///{os.path.abspath(self.optuna_dir)}/{self.experiment_name}.db"

        if self.sampler == 'random':
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        elif self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=self.seed)

        study = optuna.create_study(direction="minimize", 
                                    study_name=self.experiment_name, 
                                    storage=storage_name, 
                                    sampler=sampler, 
                                    load_if_exists=True)
        
        study.optimize(self.objective, n_trials=self.n_trials)


class HyperOpt_StdAnneal_LR(BaseExperiment):
    def __init__(self, params, find_lr=False):
        
        """
        This experiment is designed to tune the loss hyperparameters (weights and discount factor) of a model
        that uses standard annealing for the loss weights and cyclical annealing for the discount factor. The model trained with annealing as opposed to 
        fixed weights has far less variance and the is less sensitive to the hyperparameters overall.

        The hyperparameters to tune are:
        - gamma n_cycles: how many times to cycle the discount factor for the state prediction loss
        - gamma ratio: what percentage of the total iterations should be spent with gamma=1.0
        - chain_rule_ratio: what percentage of the total iterations should be spent with the chain rule loss at 1.0
        - lat_pred_ratio: what percentage of the total iterations should be spent with the latent prediction loss at 1.0
        - state_pred_ratio: what percentage of the total iterations should be spent with the state prediction loss at 1.0

        Args:
            params (dict): Dictionary of parameters for the experiment.
            The parameters are:
            - experiment_name (str): Name of the experiment.
            - data_path (str): Path to the training state data.
            - seed (int): Random seed.
            - latent_dim (int): Dimension of the latent space.
            - n_delays (int): Number of delays.
            - n_trials (int): Number of trials to run.  
            - accelerator (str): Accelerator to use. 'gpu', 'tpu' or 'cpu'.
            - sampler (str): Which optuna sampler to use. 'random' or 'tpe'.
            - gpu_idx (int): Index of the GPU to use.
        """
        super().__init__(params)
        self.params = params

        self.objective = None
        self.total_epochs = self.pretrain_epochs + self.main_epochs
        self.find_lr = find_lr
        self.lr = params['lr']
        self.min_lr = params['min_lr']
                                             
    def setup(self):
        """
        This method sets up the experiment by creating the data module, model, and objective function for optuna.
        """

        seed_everything(self.seed)
        dm = DelayKoopDataModule(data_path=self.data_path,
                                  batch_size=self.batch_size, 
                                  prediction_horizon=self.val_horizon, # for the validation forward pass
                                  seed=self.seed)
        
        ae_p = cm.ae_params(decoder_hid_dim=self.decoder_hidden_dim,
                         encoder_hid_dim=self.encoder_hidden_dim,
                         encoder_n_layers=self.encoder_hidden_layers,
                         decoder_n_layers=self.decoder_hidden_layers,
                         embedding_dim=self.latent_dim,
                         dropout_rate=self.dropout
        )

        data_p = cm.data_params(n_delays=self.n_delays,
                             n_states=self.n_states,
        )



        loss_p = cm.loss_params(full_jacobian=self.params['full_jacobian'])

        def objective(trial):
            # to run a specific trial, the trial range is narrowed
            chain_rule_ratio = trial.suggest_float("chain_rule_ratio", 0.2, .8)
            lat_pred_ratio = trial.suggest_float("lat_pred_ratio", 0.2, .5)
            state_pred_ratio = lat_pred_ratio #trial.suggest_float("state_pred_ratio", 0.2, .8)
            gamma_ratio = trial.suggest_float("gamma_ratio", 0.5, .9)
            mu_ratio = trial.suggest_float("mu_ratio", 0.2, .8)
            lr = trial.suggest_float("lr", self.min_lr, self.lr)

            print(f"chain_rule_ratio: {chain_rule_ratio}")
            print(f"lat_pred_ratio: {lat_pred_ratio}")
            print(f"state_pred_ratio: {state_pred_ratio}")
            print(f"gamm_ratio: {gamma_ratio}")
            print(f"mu_ratio: {mu_ratio}")
            print(f"lr: {lr}")

            cyclical_annealing_callback = cm.CyclicalAnnealingCallback(max_epochs=self.main_epochs, 
                                                                       steps_per_epoch=dm.steps_per_epoch,
                                                                       pretrain_epochs=self.pretrain_epochs,
                                                                       n_cycle=(1, 1, 1, 1, 1), 
                                                                       ratio=(chain_rule_ratio, lat_pred_ratio, state_pred_ratio, gamma_ratio, mu_ratio), 
                                                                       annealing=True,
                                                                        reg_annealing=True, 
                                                                        type='sigmoid')
            
            training_p = cm.training_params(learning_rate=lr,
                                        steps_per_epoch=dm.steps_per_epoch,
                                        max_epochs = self.total_epochs,
                                        prediction_horizon_val=self.val_horizon, # for the validation forward pass
                                        prediction_horizon_train=self.train_horizon # for the training forward pass 
                                        )
            
            logger = TensorBoardLogger(self.log_dir, name='tb_logs')

            model = DelayKoop(ae_params=ae_p,
                            data_params=data_p,
                            loss_params=loss_p,
                            training_params=training_p)

            val_loss_tracker = cm.MetricTracker()

            ckpt_callback = ModelCheckpoint(dirpath=self.checkpoint_dir,
                                            filename=f"trial_{trial.number}_cr{chain_rule_ratio}_pr{lat_pred_ratio}_lr{lr}_gr{gamma_ratio}_mr{mu_ratio}")

            trainer = Trainer(logger=logger,
                              max_epochs=self.total_epochs,
                              accelerator=self.accelerator,
                              devices=[self.gpu_idx],
                                callbacks=[cyclical_annealing_callback, val_loss_tracker, ckpt_callback],
                                check_val_every_n_epoch=1)
            if self.find_lr:
                tuner = pl.tuner.tuning.Tuner(trainer)
                lr_finder = tuner.lr_find(model, dm)
                print(lr_finder.suggestion())
                fig = lr_finder.plot(suggest=True)
                fig.savefig(f'{self.log_dir}/lr_finder.png')
            else:
                trainer.fit(model, dm)
            
            val_loss = val_loss_tracker.metric[-1]

            #if trial.number > 0:
            #    best_val_loss = trial.study.best_trial.value

            #    best_trial = trial.study.best_trial.number

            #    if val_loss < best_val_loss:
            #        os.remove(f"{self.checkpoint_dir}/best_trial_{best_trial}.ckpt")
            #    else:
            #        os.remove(f"{self.checkpoint_dir}/best_trial_{trial.number}.ckpt")
            
            return val_loss

        self.objective = objective

    def run(self):
        """
        This method runs the experiment by creating a study and running the trials.
        """
        storage_name = storage_name = f"sqlite:///{os.path.abspath(self.optuna_dir)}/{self.experiment_name}.db"

        if self.sampler == 'random':
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        elif self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=self.seed)

        study = optuna.create_study(direction="minimize", 
                                    study_name=self.experiment_name, 
                                    storage=storage_name, 
                                    sampler=sampler, 
                                    load_if_exists=True)
        
        study.optimize(self.objective, n_trials=self.n_trials)