# Deep Linear Time-delayed Dynamics

## Logging
The logs are saved in the `logs` folder. The logs are saved in the following format:
```
 logs\{dataset_name}_{experiment_name}_{seed}
```
Inside each `{dataset_name}_{experiment_name}_{seed}` folder, the contents are:
```
\{dataset_name}_{experiment_name}_{seed}
    \tb_logs
        \version_0  
        \version_1
        ...
        \version_n # Tensorboard logs for n trials
    \checkpoints
        \best_trial_{trial.number}.ckpt  Checkpoint for the best model
    \optuna
        \study.db  # Optuna study database
    \csv
        \main_results.csv # Main Results from the Tensorboard logs and the Optuna study database
```