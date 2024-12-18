<h1 align="center">Automated Global Analysis of Experimental Dynamics through Low-Dimensional Linear Embeddings</h1>


[Sam Moore](https://samavmoore.github.io/), [Brian Mann](https://mems.duke.edu/people/brian-mann/), [Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>
[arXiv](https://arxiv.org/abs/2411.00989)
[Project Page](http://generalroboticslab.com/AutomatedGlobalAnalysis)
[Video](https://www.youtube.com/watch?v=8Q5NQegHz50)
<br>
## Overview
Dynamical systems theory has long provided a foundation for understanding evolving phenomena across scientific domains. Yet, the application of this theory to complex real-world systems remains challenging due to issues in mathematical modeling, nonlinearity, and high dimensionality. In this work, we introduce a data-driven computational framework to derive low-dimensional linear models for nonlinear dynamical systems directly from raw experimental data. This framework enables global stability analysis through interpretable linear models that capture the underlying system structure. Our approach employs time-delay embedding, physics-informed deep autoencoders, and annealing-based regularization to identify novel low-dimensional coordinate representations, unlocking insights across a variety of simulated and previously unstudied experimental dynamical systems. These new coordinate representations enable accurate long-horizon predictions and automatic identification of intricate invariant sets while providing empirical stability guarantees. Our method offers a promising pathway to analyze complex dynamical behaviors across fields such as physics, climate science, and engineering, with broad implications for understanding nonlinear systems in the real world.
<div style="text-align: center;">
  <img src="./DelayKoop/make_gif_high_res.gif" alt="make_gif_high_res" width="600">
</div>

## Contents
- [Prerequisites](#prerequisites)
- [Training](#training)
- [Logging](#logging)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
## Prerequisites
### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```
### Datasets
Make a directory called `Data` in the root directory of the repository.
```bash
mkdir Data
```
The datasets are saved in here as:
```
 Data\{system_name_from_config.yaml}\{dataset_name_with_hyperparam_values}.npy
```
You can generate your own simulated datasets by using the GenerateDataset class in generate_datasets.py after providing the config path.
```
config_yaml_path = './DelayKoop/Datasets/Configs/lorenz_96_configs.yaml'
dataset = GenerateDataset(config_yaml_path)
dataset.collect_data()
```
The experimental datasets are provided directly in this repository as CSVs. You can process and save the datasets in the `Data` dir by running
``` bash
python -m DelayKoop.Datasets.process_doub_pend
python -m DelayKoop.Datasets.process_mag_pend
```
You can also download the simulated datasets from the following link:
- [Data](url)

## Training
Once the datasets are saved in the `Data` directory, you can train the models on the desired dataset by adding the data path to the params in `DelayKoop/Experiments/run_all.py`. This file contains the configurations for the experiments in the paper. They were trained using a class that performs hyperparameter search with optuna over the annealing params and the learning rate `HyperOpt_StdAnneal_LR`, these experiments can be run using the experiments runner object `Experiment_Runner`. For example, to train the model on the Lorenz-96 dataset:
```
params = params_default
params['n_trials'] = 12
params['data_path'] = "./Data/Lorenz_96_Sim/dataset_name"
params['latent_dim'] = 14
params['n_delays'] = 20
params['train_horizon'] = 220
params['val_horizon'] = 270
params['downsample_fctr'] = 2
params['batch_size'] = 128
params['main_epochs'] = 100
params['n_states'] = 40
params['experiment_name'] = '14dim'
params['lr'] = 1e-3

runner = ExperimentRunner()
exp = HyperOpt_StdAnneal_LR(params=params)
runner.add_experiment(exp)
runner.run_all()
```
You can directly run the experiments in the `run_all.py` file by running:
```bash
python -m DelayKoop.Experiments.run_all
```

## Logging
The logs are saved in the `logs` folder in the following format:
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

## Citation
If you find this work useful, please cite our paper:
```arxiv
@misc{moore2024automatedglobalanalysisexperimental,
      title={Automated Global Analysis of Experimental Dynamics through Low-Dimensional Linear Embeddings}, 
      author={Samuel A. Moore and Brian P. Mann and Boyuan Chen},
      year={2024},
      eprint={2411.00989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.00989}, 
}  
```
## Acknowledgements
This work was supported by the National Science Foundation Graduate Research Fellowship, the ARL STRONG program under awards W911NF2320182
and W911NF2220113, by ARO W911NF2410405, by DARPA FoundSci program under
award HR00112490372, and DARPA TIAMAT program under award HR00112490419.
