# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torchinfo import summary

from dataset import DataSetCreation

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class CNN_classifier:
    """Class implementing Convolutional Neural Network classifier supporting single or multiple hyperparameter sets."""

    def __init__(self, config):
        """
        ...
        """

        self.config = config
        
        # Import dataset_path from configuration file
        dataset_path = config.get("dataset_path")
        self.csv_path = Path(dataset_path)
        if self.csv_path is None:
            raise ValueError("dataset_path must be specified in the YAML configuration.")

        # Define default CNN hyperparameters and update them if other inputs are given
        default_params = {
            'conv_filters': [16, 32],
            'kernel_size': 5,
            'pooling_size' : 2,
            'dropout_rates_1': [0.2, 0.2],
            'dropout_rates_2': [0.3],
            'hidden_dims': [32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
        }

        # Import hyperparameters of the CNN model from the configuration file
        raw_params = config.get("CNN_params")
        self.param_configs = []

        # Handle whether CNN_params is a single dict, a list of dicts, or None
        if raw_params is None or raw_params == {}:  # No hyperparameters in input
            self.param_configs.append(default_params)
        elif isinstance(raw_params, list):  # List of different sets of hyperparameters
            self.param_configs = []
            for params in raw_params:
                p = default_params.copy()
                p.update(params)
                self.param_configs.append(p)
        else:
            default_params = default_params.copy()  # Single set of hyperparameters
            if raw_params is not None:
                default_params.update(raw_params)
            self.CNN_params = default_params
            self.param_configs = [default_params]

        # Select computer device to be used
        device_config = config.get('device', None)
        if device_config:
            self.device = torch.device(device_config)
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        print(f"Computer device used: {self.device}")

        # Initialize the PyTorch CNN model with the first configuration and send it to the available device
        self.load_data()
        self.model, self.optimizer, self.loss_function = self.build_cnn(self.param_configs[0])

    def load_data(self):
        """Load data properly formatted by calling the DataSetCreation class from dataset.py in "deep_learning" mode."""
        loader = DataSetCreation(csv_path=self.csv_path, mode="deep_learning")
        self.train, self.test, self.validation = loader.load_pytorch_data()
        return self.train, self.test, self.validation

    def build_cnn(self, config_params=None):
        """..."""

        # Fall back to the first configuration if None is provided
        if config_params is None:
            config_params = self.param_configs[0]

        # Extract network parameters
        conv_filters = config_params['conv_filters']
        kernel_size = config_params['kernel_size']
        pooling_size = config_params['pooling_size']
        dropout_rates = config_params['dropout_rates_1']
        dropout_rates2 = config_params['dropout_rates_2']
        hidden_dims = config_params['hidden_dims']
        learning_rate = config_params['learning_rate']
        batch_size = config_params['batch_size']

        # Extracts automatically input channels from the loaded trained dataset
        sample_x, _ = self.train[0]
        in_channels = sample_x.shape[0]

        # FEATURE EXTRACTOR (convolutional layers)
        layers = []
        current_channels = in_channels
        
        # Loop dynamically through every filter size specified in the configuration file
        for out_channels, dr_rate in zip(conv_filters, dropout_rates):
            layers.append(nn.Conv1d(in_channels=current_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pooling_size))
            layers.append(nn.Dropout(dr_rate))
            
            # The output channels of this layer become the input channels of the next layer
            current_channels = out_channels

        # Turn the list of layers into a nn.Sequential container
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)

        # CLASSIFICATION HEAD (dense linear layers)
        input_dim = conv_filters[-1]
        dense_layers = [nn.Flatten()]  # Flatten the output from feature extractor into a 1D object

        for h_dim, dr_rate2 in zip(hidden_dims, dropout_rates2):
            dense_layers.append(nn.Linear(in_features=input_dim, out_features=h_dim))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dr_rate2))
            input_dim = h_dim

        # Final output layer for binary classification
        dense_layers.append(nn.Linear(in_features=input_dim, out_features=1))
        self.classifier = nn.Sequential(*dense_layers)

        # MODEL, OPTIMIZER AND LOSS FUNCTION DEFINITION
        self.model = nn.Sequential(self.features, self.classifier).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.BCEWithLogitsLoss()

        # Print a summary of the defined network
        #print(summary(self.model, input_size=(batch_size, sample_x.shape[0], sample_x.shape[1]), col_names=("output_size", "num_params"), depth=3))

        return self.model, self.optimizer, self.loss_function
    

    #def train_model():

    #def evaluate_with_optimal_threshold():

    #def compute_confusion_matrix():

    def run_all_configurations(self):
        """
        Iterate through all loaded configurations of hyperparameters using the same data split.
        
        Returns
        -------
        results : list of tuples
            A list where each element contains a tuple of `(params, conf_matrix, metrics)` for each evaluated 
            hyperparameter configuration.
         """
        
        results = []
        for i, params in enumerate(self.param_configs, start=1):
            print(f"\n==============================")
            print(f"Running Configuration {i}: {params}")
            print(f" ")
                    
            # Re-initialize the model with this specific configuration of hyperparameters
            self.model, self.optimizer, self.loss_function = self.build_cnn(params)
            #self.train_model()
            #conf_matrix, metrics = self.compute_confusion_matrix()
            #results.append((params, conf_matrix, metrics))
    
        
        return None

if __name__ == "__main__":

    # Import data and files of parameters
    csv_path = Path(__file__).resolve().with_name("tess_data.csv")
    config = {"dataset_path": csv_path}

    # Initialize CNN Classifier and call this method
    cnn_classifier = CNN_classifier(config=config)
    results_all = cnn_classifier.run_all_configurations()