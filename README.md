<h1 align="center">Automated Global Analysis of Experimental Dynamics through Low-Dimensional Linear Embeddings</h1>


[Sam Moore](https://samavmoore.github.io/), [Brian Mann](https://mems.duke.edu/people/brian-mann/), [Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>
## Overview
Dynamical systems theory helps scientists and engineers understand changing phenomena in every corner of study. However, applying this theoretical framework to understand real world systems remains challenging. This difficulty arises, in part, from issues in mathematical modeling, nonlinearity, and dimensionality. Our work addresses these challenges with a computational pipeline based on deep learning to find low-dimensional linear models for nonlinear dynamics directly from experimental data. Moreover, these new linear models possess a structure that can be easily interpreted and exploited to perform global analysis of the system’s stability behavior. To find these models, our framework uses time-delay embedding, physics-informed deep autoencoders, and annealing-based regularization. With our method, we discover new, low-dimensional, coordinate representations for a wide range of simulated and previously unstudied experimental dynamical systems across scientific fields. We show that these new coordinate representations, for all studied systems,  achieve accurate long-horizon predictions and automatically uncover intricate invariant sets while providing empirical stability guarantees.
<div style="text-align: center;">
  <img src="./DelayKoop/linearization.gif" alt="Linearization" width="600">
</div>

## Contents
- [Prerequisites](#prerequisites)
- [Training](#training)
- [Logging](#logging)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
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
        \main_results.csv # Main results from the Tensorboard logs and the Optuna study database
```