import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from .networks import Encoder, Decoder
from torch.optim.lr_scheduler import OneCycleLR
from .common import ae_params, data_params, loss_params, training_params
from torch import nn
from functorch import jacrev

######################## ------------------------------ Main Lightning Module ------------------------------############################


class DelayKoop(LightningModule):
    def __init__(self,
                  ae_params=ae_params(),
                    data_params=data_params(),
                    loss_params=loss_params(),
                    training_params=training_params(), 
                 ): 
        super().__init__()
        """
        Initializes the DelayKoop model

        Args:

        ae_params: namedtuple containing the parameters for the autoencoder architecture
        data_params: namedtuple containing the parameters for the data
        loss_params: namedtuple containing the parameters for the loss function
        training_params: namedtuple containing the parameters for the training procedure

        """

        self.save_hyperparameters()

        # initialize model
        self.n_states = data_params.n_states # number of states in the system
        self.n_delays = data_params.n_delays # number of delays in the system
        self.delta_t = data_params.delta_t # time step
        self.embedding_dim = ae_params.embedding_dim # dimension of the embedding

        self.input_dim = self.n_states * self.n_delays


        self.encoder = Encoder(n_hid_layers=ae_params.encoder_n_layers, hidden_dim=ae_params.encoder_hid_dim,
                                batch_norm=ae_params.batch_norm, res_net=ae_params.res_net, input_dim=self.input_dim, 
                                output_dim=self.embedding_dim, dropout_rate=ae_params.dropout_rate)


        self.decoder = Decoder(n_hid_layers=ae_params.decoder_n_layers, hidden_dim=ae_params.decoder_hid_dim,
                                    batch_norm=ae_params.batch_norm, res_net=ae_params.res_net, input_dim=self.embedding_dim, 
                                    output_dim=self.n_states, dropout_rate=ae_params.dropout_rate)

        # initialize the Koopman generator G as a linear layer with no bias
        self.G = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)

        self.chain_rule_loss_wt = loss_params.chain_rule_loss_wt
        self.pred_loss_wt = loss_params.pred_loss_wt
        self.recon_loss_wt = loss_params.recon_loss_wt
        self.state_pred_loss_wt = loss_params.state_pred_loss_wt
        self.gamma = loss_params.gamma
        self.G_reg_wt = loss_params.G_reg_wt
        self.full_jacobian = loss_params.full_jacobian

        self.learning_rate = training_params.learning_rate
        self.steps_per_epoch = training_params.steps_per_epoch
        self.max_epochs = training_params.max_epochs

        self.train_prediction_horizon = training_params.prediction_horizon_train
        self.val_prediction_horizon = training_params.prediction_horizon_val

        self.downsample_factor = training_params.downsample_factor

    def forward(self, x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf, mode='train'):
        """
        Runs the model on a batch of data

        Args:

        x_t0: initial state with time delays (batch_size, input_dim)
        fx_t0: initial dynamics/acceleration with time delays (batch_size, input_dim)
        x_tn: ground truth future/past trajectory (batch_size, input_dim, prediction_horizon_val)
        x_tf: final state (batch_size, input_dim)
        fx_tf: final dynamics/acceleration (batch_size, input_dim)
        mode: whether to use the training or validation prediction horizon (default: 'train')
        batch_idx: index of the batch (default: 0)
        """
        if mode == 'train':
            prediction_horizon = self.train_prediction_horizon
        elif mode == 'val':
            prediction_horizon = self.val_prediction_horizon
            prediction_horizon -= 1
        else:
            raise ValueError("mode must be either 'train' or 'val'")

        outputs = {}
        
        x_tn = x_tn[:, :, :prediction_horizon]
        x_tn = x_tn[:, :, ::self.downsample_factor]

        # embed the initial state
        psi_t0 = self.encoder(x_t0)
        outputs['psi_t0'] = psi_t0

        # reconstruct the initial state
        outputs['x_t0_hat'] = self.decoder(psi_t0)

        # embed the ground truth future trajectory
        outputs['psi_tn'] = self.true_psi_tn(x_tn)
    
        outputs['G_dot_psi_t0'] = self.G(psi_t0)
    
        # Get the Jacobian of the initial embedding with respect to the initial state
        J_t0 = self.get_batch_jacobian(x_t0)
        if not self.full_jacobian:
            J_t0 = J_t0[:, :, ::self.n_delays]
            fx_t0 = fx_t0[:, ::self.n_delays]

        # Multiply the Jacobian by the initial dynamics (velocity and accel) for the other chain rule loss term
        outputs['J_dot_fx_t0'] = torch.bmm(J_t0, fx_t0.unsqueeze(-1)).squeeze(-1)
    
        # predict future embeddings with forward integration in time
        outputs['psi_tn_hat'] = self.pred_psi_tn(psi_t0, t0, tn, prediction_horizon)
        outputs['x_tn_hat'] = self.pred_x_tn(outputs['psi_tn_hat'])


        outputs['x_tn'] = x_tn

        return outputs
        
    def training_step(self, batch, batch_idx):
        """
        Computes the loss function for a batch of data
        Logs the various loss terms and the total loss

        Args:

        batch: batch of data
        batch_idx: index of the batch
        """
        x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf = batch
        

        preds = self(x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf, mode='train')

        # get the reconstructed initial state
        x_t0_hat = preds['x_t0_hat']

        # get the initial embedding times G for the chain rule loss term
        G_dot_psi_t0 = preds['G_dot_psi_t0']

        # get the initial dynamics times the Jacobian of the encoder (psi) for the other chain rule loss term
        J_dot_fx_t0 = preds['J_dot_fx_t0']

        # compute the chain rule loss
        chain_rule_loss = F.mse_loss(G_dot_psi_t0, J_dot_fx_t0)

        x_tn = preds['x_tn']
        # the auxillary data is not used in the loss function
        x_t0 = x_t0[:, :self.n_states*self.n_delays]


        x_t0 = x_t0[:, ::self.n_delays]
        x_tn = x_tn[:, ::self.n_delays, :]

        # compute the reconstruction loss
        if self.HodHux:
            recon_loss = self.mahalanobis_mse_loss(x_t0, x_t0_hat)
        else:
            recon_loss = F.mse_loss(x_t0, x_t0_hat)

        # get the ground truth future embeddings
        psi_tn = preds['psi_tn']

        # get the predicted future embeddings
        psi_tn_hat = preds['psi_tn_hat']
        
        # compute the prediction loss
        pred_loss = self.discounted_mse(psi_tn, psi_tn_hat)

        # Long term prediction loss in state space
        x_tn_hat = preds['x_tn_hat']
        if self.HodHux:
            state_space_loss = self.discounted_mahalanobis_mse_loss(x_tn, x_tn_hat)
        else:
            state_space_loss = self.discounted_mse(x_tn, x_tn_hat)

        full_state_space_loss = F.mse_loss(x_tn, x_tn_hat)
        
        eig_clamp, n_pos_eigs = self.G_reg()

        total_loss =  self.chain_rule_loss_wt*(chain_rule_loss) + self.pred_loss_wt*(pred_loss) + self.recon_loss_wt*(recon_loss) + self.state_pred_loss_wt*(state_space_loss) + self.G_reg_wt*eig_clamp
        
        logs = {}
        logs['pred'] = pred_loss
        logs['recon'] = recon_loss
        logs['cr'] = chain_rule_loss
        logs['dsct_ss_pred'] = state_space_loss
        logs['ss_pred'] = full_state_space_loss
        logs['eig_clamp'] = eig_clamp
        logs['n_pos_eigs'] = n_pos_eigs
        logs['training'] = total_loss
    
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Computes the loss function for a batch of data
        Logs the various loss terms and the total loss

        Args:

        batch: batch of data
        batch_idx: index of the batch
        """
        x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf = batch
        

        preds = self(x_t0, fx_t0, x_tn, x_tf, fx_tf, t0, tn, tf, mode='train')

        # get the reconstructed initial state
        x_t0_hat = preds['x_t0_hat']

        # get the initial embedding times G for the chain rule loss term
        G_dot_psi_t0 = preds['G_dot_psi_t0']

        # get the initial dynamics times the Jacobian of the encoder (psi) for the other chain rule loss term
        J_dot_fx_t0 = preds['J_dot_fx_t0']

        # compute the chain rule loss
        chain_rule_loss = F.mse_loss(G_dot_psi_t0, J_dot_fx_t0)

        x_tn = preds['x_tn']
        # the auxillary data is not used in the loss function
        x_t0 = x_t0[:, :self.n_states*self.n_delays]


        x_t0 = x_t0[:, ::self.n_delays]
        x_tn = x_tn[:, ::self.n_delays, :]

        # compute the reconstruction loss
        recon_loss = F.mse_loss(x_t0, x_t0_hat)

        # get the ground truth future embeddings
        psi_tn = preds['psi_tn']

        # get the predicted future embeddings
        psi_tn_hat = preds['psi_tn_hat']
        
        # compute the prediction loss
        pred_loss = self.discounted_mse(psi_tn, psi_tn_hat)

        # Long term prediction loss in state space
        x_tn_hat = preds['x_tn_hat']
        state_space_loss = self.discounted_mse(x_tn, x_tn_hat)

        full_state_space_loss = F.mse_loss(x_tn, x_tn_hat)
        
        eig_clamp, n_pos_eigs = self.G_reg()

        total_loss =  self.chain_rule_loss_wt*(chain_rule_loss) + self.pred_loss_wt*(pred_loss) + self.recon_loss_wt*(recon_loss) + self.state_pred_loss_wt*(state_space_loss) + self.G_reg_wt*eig_clamp
        
        logs = {}
        logs['val_pred'] = pred_loss
        logs['val_recon'] = recon_loss 
        logs['val_cr'] = chain_rule_loss
        logs['validation'] = total_loss
        logs['val_ss_pred'] = full_state_space_loss
    
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def G_reg(self):
        eigs  = torch.linalg.eigvals(self.G.weight)

        r = torch.real(eigs)

        positives = torch.clamp(r, min=0)

        n_pos = torch.count_nonzero(positives)

        return torch.sum(positives), n_pos
    
    def discounted_mse(self, y_true, y_pred):
        """
        Computes the discounted MSE loss between (arbitrary named) y_true and y_pred along the 3rd dimension (time)

        Args:

        y_true: ground truth future trajectory (batch_size, embedding_dim, n_steps)
        y_pred: predicted future trajectory (batch_size, embedding_dim, n_steps)
        gamma: discount factor
        """   
        n_steps = y_pred.shape[2]
        weights = self.gamma ** torch.arange(n_steps, dtype=torch.float32).to(y_true.device)  # this creates a sequence: 1, gamma, gamma^2, ..., gamma^(n_steps-1)
        #weights = weights / torch.sum(weights)  # normalize weights to sum to 1
        weights = weights.unsqueeze(0).unsqueeze(1)
        dsct_loss = torch.mean(weights * torch.square(y_true - y_pred))
        return dsct_loss
    
    def discounted_mahalanobis_mse_loss(self, target, input):
        """
        Computes the discounted Mahalanobis MSE loss between (arbitrary named) y_true and y_pred along the 3rd dimension (time)

        Args:

        y_true: ground truth future trajectory (batch_size, embedding_dim, n_steps)
        y_pred: predicted future trajectory (batch_size, embedding_dim, n_steps)
        gamma: discount factor
        """   
        n_steps = input.shape[2]
        weights = self.gamma ** torch.arange(n_steps, dtype=torch.float32).to(input.device)
        dsct_loss = torch.mean(weights * self.mahalanobis_mse_loss(target, input))
        return dsct_loss

    def mahalanobis_mse_loss(self, target, input):
        # Calculate the variance along the feature dimension of the target
        variance = torch.var(target, dim=0, unbiased=False)
        
        # Ensure the variance is not zero to avoid division by zero
        variance[variance == 0] = 1e-6

        # Compute the Mahalanobis distance assuming the covariance matrix is diagonal
        diff = input - target
        dist = (diff ** 2) / variance
        if len(dist.shape) == 3:
            mse = torch.mean(dist, dim=0)
            mse = torch.mean(mse, dim=0)
        else:
            mse = torch.mean(dist)
        return mse

    def get_batch_jacobian(self, x):
        """
        Computes the Jacobian of the encoder with respect to the input x
        x: input to the encoder (batch_size, n_states*n_delays+aux_dim)
        """
        jac_vmap = torch.vmap(jacrev(self.encoder), randomness='different')(x)

        return jac_vmap
    
    # define l2 regularization on the weights
    def l2_reg(self, params):
        sum = 0
        for name, param in params:
            if "weight" in name and param.requires_grad:
                sum += torch.linalg.norm(param)**2
        return sum
    
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(optimizer, self.learning_rate, epochs=self.max_epochs, steps_per_epoch=self.steps_per_epoch)
        outs = [{"optimizer": optimizer, "lr_scheduler": { "scheduler": scheduler, "interval": "step"}}]
        return outs
    

    def pred_psi_tn(self, psi_t0, t0, tn, horizon=None):
        """
        Predicts the future embedding trajectory given the initial or final embedding (psi_t0 is slightly abused here casue it can be either the initial or final embedding)
        Gives e^Gt*psi_t0 for t = 0, ..., prediction_horizon*delta_tf 
        psi_x: initial embedding (batch_size, embedding_dim)

        """
        assert horizon is not None, "Must specify horizon"
        batch_size, n_embeddings = psi_t0.shape
        psi_tn_hat = torch.zeros((batch_size, self.embedding_dim, horizon), device=self.device)


        G = self.G.weight
        
        t = torch.sub(tn[:, 0, :], t0[:, 0].unsqueeze(-1)) #torch.linspace(self.delta_t, (horizon)*self.delta_t, horizon, device=self.device)

        t = t[:, :horizon]
        t = t[:, ::self.downsample_factor]
        psi_tn_hat = psi_tn_hat[:, :, ::self.downsample_factor]

        G_repeat = G.expand(t.shape[0], t.shape[1], G.shape[0], G.shape[1])
        ts = t.unsqueeze(-1).unsqueeze(-1)
        G_t = G_repeat * ts
        exp_G = torch.matrix_exp(G_t)

        psi_t0s = psi_t0#.unsqueeze(1)
        #psi_t0s = psi_t0s.transpose(0, 1)
        #psi_t0_repeat = psi_t0s.repeat(1, 1, psi_tn_hat.shape[2])
        #psi_t0_repeat = psi_t0_repeat.transpose(0, 2)

        psi_tn_hat = torch.vmap(lambda x, y: torch.matmul(x, y))(exp_G, psi_t0s)
        #psi_tn_hat = torch.matmul(exp_G, psi_t0s)
        #psi_tn_hat = psi_tn_hat.transpose(0, 2)

        return psi_tn_hat.transpose(1, 2)


    def true_psi_tn(self, x):
        ''' 
        Computes the true future embedding trajectory given the actual future states
        Gives psi_tn for t = 0, ..., prediction_horizon*delta_tf
        x: future states (batch_size, n_states, n_pred_steps)
        '''
        batch_size, n_inputs, horizon = x.shape
        x_reshaped = x.transpose(1, 2).contiguous().view(-1, n_inputs)
        psi_tn = self.encoder(x_reshaped)
        psi_tn = psi_tn.view(batch_size, horizon, -1).transpose(1, 2)

        return psi_tn
    
    def pred_x_tn(self, pred_psi_tn):
        """
        Reconstructs a trajectory in state space given a predicted embedding trajectory
        -in other words this function decodes the predicted embedding trajectory
        
        pred_psi_tn: predicted embedding trajectory (batch_size, embedding_dim, n_pred_steps)
        """
        batch_size, n_embeddings, horizon = pred_psi_tn.shape
        psi_tn_reshaped = pred_psi_tn.transpose(1, 2).contiguous().view(-1, n_embeddings)
        x_tn_hat = self.decoder(psi_tn_reshaped)
        x_tn_hat = x_tn_hat.view(batch_size, horizon, -1).transpose(1, 2)

        return x_tn_hat