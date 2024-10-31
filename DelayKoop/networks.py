import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim=30, 
                 output_dim=30, 
                 hidden_dim=128, 
                 n_hid_layers=3, 
                 batch_norm=True, 
                 res_net=True, 
                 dropout_rate=None
                 ):
        
        super().__init__()

        # Initializing the components of the encoder
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))

        # Initializing batch normalization and residual connections options
        self.batch_norm = batch_norm
        self.res_net = res_net

        # Batch normalization after the input layer
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        # Creating the hidden layers
        for _ in range(n_hid_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                # Batch normalization for each hidden layer
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Forward function will define the forward pass of the input through the network
        if self.res_net:
            # If residual connections are enabled
            
            # Start from the first layer
            x_temp = self.model[0](x)
            x_out = x_temp

            # Apply remaining layers
            for i, layer in enumerate(self.model[1:], 1):
                if isinstance(layer, nn.BatchNorm1d):
                    x_temp = layer(x_temp)
                else:
                    x_temp = layer(F.relu(x_temp))
                
                # Add residual connection if not the last layer
                if i != len(self.model) - 1:
                    x_out = x_out + x_temp
                else:
                    x_out = x_temp
        else:
            # If residual connections are not enabled, just apply the model to the input
            x_out = self.model(x)

        return x_out


class Decoder(nn.Module):
    def __init__(self, input_dim=30, 
                 output_dim=2, 
                 hidden_dim=128, 
                 n_hid_layers=3, 
                 batch_norm=True, 
                 res_net=True, 
                 dropout_rate=None):
        
        super().__init__()

        # Same structure as Encoder, but may have different dimensions
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norm = batch_norm
        self.res_net = res_net

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(n_hid_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Same structure as the forward function in the Encoder

        if self.res_net:
            x_temp = self.model[0](x)
            x_out = x_temp

            for i, layer in enumerate(self.model[1:], 1):
                if isinstance(layer, nn.BatchNorm1d):
                    x_temp = layer(x_temp)
                else:
                    x_temp = layer(F.relu(x_temp))
                
                if i != len(self.model) - 1:
                    x_out = x_out + x_temp
                else:
                    x_out = x_temp
        else:
            x_out = self.model(x)

        return x_out
