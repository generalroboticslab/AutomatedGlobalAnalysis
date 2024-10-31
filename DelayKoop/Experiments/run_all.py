from .experiments import HyperOpt_StdAnneal, HyperOpt_StdAnneal_LR 
from .base import ExperimentRunner
import threadpoolctl as tpc
import multiprocessing

params_default = {'experiment_name': 'Experiment',
                    'data_path': './data',
                    'seed': 123,
                    'latent_dim': 4,
                    'n_delays': 20,
                    'n_trials': 100,
                    'accelerator': 'gpu',
                    'gpu_idx': None,
                    'batch_size': 256,
                    'min_lr': 3e-4,
                    'pretrain_epochs': 5,
                    'main_epochs': 150,
                    'n_states': 2,
                    'train_horizon': 400,
                    'val_horizon': 450,
                    'downsample_fctr': 1,
                    'encoder_hidden_dim': 256,
                    'decoder_hidden_dim': 256,
                    'encoder_hidden_layers': 3,
                    'decoder_hidden_layers': 3,
                    'dropout': 0.001,
                    'full_jacobian': True
                    }

if __name__ == "__main__":
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))
    with tpc.threadpool_limits(limits=limits):

        runner = ExperimentRunner()
        ##### ---------- Magnetic Mass Spring Damper ---------- #####

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/Linear_Magnet_Sim/linear_magnet_model_delaytype-dt_delaysteps-41_stepsize-0.04_noise-0_ntraj-800_trajlen-1000.npy"
        params['n_delays'] = 41
        params['gpu_idx'] = 0
        params['train_horizon'] = 400
        params['val_horizon'] = 480
        params['downsample_fctr'] = 2
        params['main_epochs'] = 130
        params['pretrain_epochs'] = 8
        params['batch_size'] = 1200
        params['n_trials'] = 12
        params['full_jacobian'] = True
        params['lr'] = 5e-3

        params['latent_dim'] = 2
        params['experiment_name'] = '2dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = '3dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = '4dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = '5dim'
        #exp = HyperOpt_StdAnneal_LR(params=params) 
        #runner.add_experiment(exp)

        params['latent_dim'] = 6
        params['experiment_name'] = '6dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 7
        params['experiment_name'] = '7dim'
        #exp = HyperOpt_StdAnneal_LR(params=params) 
        #runner.add_experiment(exp)


        ##### ---------- Single Pendulum ---------- #####
        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/Pendulum_Sim/pend_delaytype-dt_delaysteps-22_stepsize-0.05_noise-0_ntraj-800_trajlen-800.npy"
        params['latent_dim'] = 2
        params['n_delays'] = 22
        params['gpu_idx'] = 0
        params['train_horizon'] = 400
        params['val_horizon'] = 450
        params['downsample_fctr'] = 2
        params['batch_size'] = 2400
        params['main_epochs'] = 100
        params['pretrain_epochs'] = 5
        params['experiment_name'] = '2dim'
        params['full_jacobian'] = True
        params['min_lr'] = 2e-4
        params['lr'] = 3e-3

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = '3dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = '4dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = '5dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)


        ##### ---------- Lorenz 96 Periodic ---------- #####``
        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "/Data/Lorenz_96_Sim/lorenz_96_delaytype-dt_delaysteps-20_stepsize-0.05_noise-0_ntraj-400_trajlen-500.npy"
        params['latent_dim'] = 2
        params['n_delays'] = 20
        params['gpu_idx'] = 0
        params['train_horizon'] = 220
        params['val_horizon'] = 270
        params['downsample_fctr'] = 2
        params['batch_size'] = 128
        params['main_epochs'] = 100
        params['n_states'] = 40
        params['full_jacobian'] = True
        params['experiment_name'] = 'lorenz-96-2dim'
        params['lr'] = 1e-3

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = 'lorenz-96-3dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = 'lorenz-96-4dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = 'lorenz-96-5dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 6
        params['experiment_name'] = 'lorenz-96-6dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 7
        params['experiment_name'] = 'lorenz-96-7dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 8
        params['experiment_name'] = 'lorenz-96-8dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 9
        params['experiment_name'] = 'lorenz-96-9dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 10
        params['experiment_name'] = 'lorenz-96-10dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 11
        params['experiment_name'] = 'lorenz-96-11dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 12
        params['experiment_name'] = 'lorenz-96-12dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 13
        params['experiment_name'] = 'lorenz-96-13dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 14
        params['experiment_name'] = 'lorenz-96-14dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 15
        params['experiment_name'] = 'lorenz-96-15dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 16
        params['experiment_name'] = 'lorenz-96-16dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 17
        params['experiment_name'] = 'lorenz-96-17dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 18
        params['experiment_name'] = 'lorenz-96-18dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/Duffing_Sim/duffing_delaytype-dt_delaysteps-21_stepsize-0.05_noise-0_ntraj-600_trajlen-1000.npy"
        params['latent_dim'] = 2
        params['n_delays'] = 21
        params['gpu_idx'] = 0
        params['train_horizon'] = 400
        params['val_horizon'] = 500 #700
        params['downsample_fctr'] = 1
        params['batch_size'] = 1000
        params['main_epochs'] = 100
        params['n_states'] = 2
        params['full_jacobian'] = True
        params['experiment_name'] = 'duffing-2dim'
        params['lr'] = 5e-3

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = 'duffing-3dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = 'duffing-4dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = 'duffing-5dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 6
        params['experiment_name'] = 'duffing-6dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 7    
        params['experiment_name'] = 'duffing-7dim'
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/Van_Der_Pol_Sim/van_der_pol_delaytype-dt_delaysteps-20_stepsize-0.05_noise-0_ntraj-400_trajlen-900.npy"
        params['latent_dim'] = 2
        params['n_delays'] = 20
        params['gpu_idx'] = 0
        params['train_horizon'] = 400
        params['val_horizon'] = 480
        params['downsample_fctr'] = 2
        params['experiment_name'] = 'VDP-2dim'
        params['batch_size'] = 1000
        params['main_epochs'] = 120
        params['lr'] = 3e-3
        params['n_states'] = 2

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = 'VDP-3dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = 'VDP-4dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = 'VDP-5dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/Double_LCO_Sim/double_lco_delaytype-dt_delaysteps-20_stepsize-0.05_noise-0_ntraj-600_trajlen-1000.npy"
        params['latent_dim'] = 2
        params['n_delays'] = 20
        params['gpu_idx'] = 0
        params['train_horizon'] = 400
        params['val_horizon'] = 480
        params['downsample_fctr'] = 2
        params['experiment_name'] = 'DLCO-2dim'
        params['batch_size'] = 1000
        params['main_epochs'] = 120
        params['lr'] = 3e-3
        params['n_states'] = 2

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = 'DLCO-3dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 4
        params['experiment_name'] = 'DLCO-4dim-redo'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 5
        params['experiment_name'] = 'DLCO-5dim-redo'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)



        ##### ---------- Double Pendulum ---------- #####

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/train-exp-doub-pend-795traj-0.02dt-26tf-100delay.npy"
        params['n_delays'] = 100
        params['gpu_idx'] = 0
        params['train_horizon'] = 150
        params['val_horizon'] = 180
        params['downsample_fctr'] = 1
        params['main_epochs'] = 100
        params['pretrain_epochs'] = 5
        params['batch_size'] = 1200
        params['full_jacobian'] = True
        params['min_lr'] = 1e-3
        params['lr'] = 3e-3
        params['latent_dim'] = 2
        params['experiment_name'] = '2-dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 3
        params['experiment_name'] = '3-dim'
        
        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)

        params['latent_dim'] = 6
        params['experiment_name'] = '6-dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)
        
        
        ##### ---------- Lorenz 96 Chaotic ---------- #####

        #params = params_default
        #params['n_trials'] = 12
        #params['sampler'] = 'tpe'
        #params['data_path'] = "./Data/data-lorenz-96-1250traj-0.02dt-20tf-41delay.npy"
        #params['latent_dim'] = 400
        #params['n_delays'] = 41
        #params['experiment_name'] = '400dim'

        #exp = Lorenz96_HyperOpt_LRTune(params=params)
        #runner.add_experiment(exp)

        ##### ---------- Magnetic Pendulum ---------- #####

        params = params_default
        params['n_trials'] = 12
        params['sampler'] = 'tpe'
        params['data_path'] = "./Data/train-mag-pend-900traj-0.02dt-10tf-31delay.npy"
        params['n_delays'] = 31
        params['gpu_idx'] = 0
        params['train_horizon'] = 300
        params['val_horizon'] = 330
        params['downsample_fctr'] = 2
        params['main_epochs'] = 200
        params['pretrain_epochs'] = 20
        params['batch_size'] = 500
        params['full_jacobian'] = True
        params['lr'] = 3e-3
        params['latent_dim'] = 2
        params['experiment_name'] = '2-dim'

        #exp = HyperOpt_StdAnneal_LR(params=params)
        #runner.add_experiment(exp)



        runner.run_all()

