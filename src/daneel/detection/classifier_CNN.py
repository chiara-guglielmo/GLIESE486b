# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gc

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import DataSetCreation

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Definition of a class implementing CNN classifier
class CNN_classifier:
    """Class implementing CNN classifier supporting single or multiple hyperparameter sets."""

    def __init__(self, config):
        """
        Initialize the CNN classifier configuration and load the dataset.

        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from the YAML parameters file, containing:
            - dataset_path : str or path-like
                Path to the input CSV file containing the data. Required.
            - device : str, optional
                Computer deviced to be used to run the classifier (default is none, which uses the 
                available device among 'cuda:0', 'mps' or evenually 'cpu').
            - CNN_params : dict or list of dict, optional
                Hyperparameters for the CNN classifier. If a dictionary is provided, 
                it contains a single configuration. If a list of dictionaries is provided, 
                multiple configurations will be evaluated (default is None, which uses default 
                hyperparameters).
        
        """

        self.config = config
        
        # Import dataset_path from configuration file
        dataset_path = config.get("dataset_path")
        self.csv_path = Path(dataset_path)
        if self.csv_path is None:
            raise ValueError("dataset_path must be specified in the YAML configuration.")

        # Define default CNN hyperparameters and update them if other inputs are given
        default_params = {
            'conv_filters': [32, 64, 128],
            'kernel_size': 7,
            'pooling_size' : 2,
            'pooling_output_size' : 8,
            'dropout_rates_1': [0.3, 0.4, 0.4],
            'dropout_rates_2': [0.4, 0.4],
            'hidden_dims': [128, 64],
            'learning_rate': 2e-4,
            'batch_size': 64,
            'epochs': 200,
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
        self.model, self.optimizer, self.scheduler, self.loss_function = self.build_cnn(self.param_configs[0])

    def load_data(self):
        """Load data properly formatted by calling the DataSetCreation class from dataset.py in "deep_learning" mode."""
        loader = DataSetCreation(csv_path=self.csv_path, mode="deep_learning")
        self.train, self.test, self.validation = loader.load_pytorch_data()
        return self.train, self.test, self.validation

    def build_cnn(self, config_params=None):
        """
        Build the 1D CNN architecture, optimizer, learning rate scheduler, and loss function. Constructs a sequential 
        1D convolutional feature extractor (Conv1d, BatchNorm1d, ReLU, MaxPool1d, Dropout, and AdaptiveAvgPool1d) followed 
        by a dense classification head for binary classification, and moves the model to 'self.device'.

        Parameters
        ----------
        config_params : dict, optional
            Hyperparameter dictionary for the network. If None, defaults to 'self.param_configs[0]'. It contains:
            - 'conv_filters' : list of int
                Number of output channels for each 1D convolutional block.
            - 'kernel_size' : int
                Kernel size used in each 1D convolutional layer.
            - 'pooling_size' : int
                Window size used in each 1D max-pooling layer.
            - 'pooling_output_size' : int
                Target sequence length produced by AdaptiveAvgPool1d.
            - 'dropout_rates_1' : list of float
                Dropout probabilities after each convolutional block.
            - 'dropout_rates_2' : list of float
                Dropout probabilities after each hidden dense layer.
            - 'hidden_dims' : list of int
                Number of units in each hidden dense layer of the classification head.
            - 'learning_rate' : float
                Initial learning rate for the Adam optimizer.
            - 'batch_size' : int
                Mini-batch size used during training and evaluation.
            - 'epochs' : int
                Number of training epochs.
        
        Returns
        -------
        model : torch.nn.Sequential
            The assembled CNN model placed on 'self.device'.
        optimizer : torch.optim.Adam
            Adam optimizer initialized with the model's parameters and configured learning rate.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            Learning rate scheduler configured to monitor a maximization metric.
        loss_function : torch.nn.BCEWithLogitsLoss
            Binary cross-entropy loss function with logits.

        Notes
        -----
        In order for the program to work, the lists 'conv_filters' and 'dropout_rates_1', 'hidden_dims' and 'dropout_rates_2'
        must have the same number of elements, since a dropout rate for each convolutional block or hidden dense layer is required.
        """

        # Fall back to the first configuration if None is provided
        if config_params is None:
            config_params = self.param_configs[0]

        # Extract network parameters
        conv_filters = config_params['conv_filters']
        kernel_size = config_params['kernel_size']
        pooling_size = config_params['pooling_size']
        pooling_output_size = config_params["pooling_output_size"]
        dropout_rates = config_params['dropout_rates_1']
        dropout_rates2 = config_params['dropout_rates_2']
        hidden_dims = config_params['hidden_dims']
        learning_rate = float(config_params['learning_rate'])
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
        layers.append(nn.AdaptiveAvgPool1d(pooling_output_size))
        self.features = nn.Sequential(*layers)

        # CLASSIFICATION HEAD (dense linear layers)
        input_dim = conv_filters[-1] * pooling_output_size
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)
        self.loss_function = nn.BCEWithLogitsLoss()

        # Print a summary of the defined network
        #print(summary(self.model, input_size=(batch_size, sample_x.shape[0], sample_x.shape[1]), col_names=("output_size", "num_params"), depth=3))

        return self.model, self.optimizer, self.scheduler, self.loss_function
    
    def train_per_epoch(self, train_loader):
        """
        Run one training epoch over the training DataLoader. It iterates through mini-batches, performs forward 
        and backward passes, updates model weights, and computes the average training loss.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader providing training feature tensors and binary labels.

        Returns
        -------
        train_loss : float
            Mean training loss across all samples in 'self.train'.
        """

        self.model.train()  # Training/Learning mode
        train_running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
        
            # Prepare inputs and layers and zero the parameter gradients
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).float().unsqueeze(1)
            self.optimizer.zero_grad()
        
            # Compute loss function, compute gradients and optimize
            loss = self.loss_function(self.model(batch_X), batch_y)
            loss.backward()
            self.optimizer.step()
            train_running_loss += loss.item() * batch_X.size(0)
        
        train_loss = train_running_loss / len(self.train)

        return train_loss
    
    def validate_per_epoch(self, validation_loader):
        """
        Evaluate the model on the validation set for one epoch without gradient updates and compute the average 
        validation loss across all validation samples.

        Parameters
        ----------
        validation_loader : torch.utils.data.DataLoader
            DataLoader providing validation feature tensors and binary labels.

        Returns
        -------
        validation_loss : float
            Mean validation loss across all samples in 'self.validation'.
        """

        self.model.eval()  # Validation/Test mode
        validation_running_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
        
                # Prepare inputs and layers
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float().unsqueeze(1)

                # Compute loss function
                loss = self.loss_function(self.model(batch_X), batch_y)
                validation_running_loss += loss.item() * batch_X.size(0)
        
            validation_loss = validation_running_loss / len(self.validation)

        return validation_loss

    def predict_probabilities(self, data_loader):
        """
        Compute predicted class probabilities and true binary labels for a certain dataset.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing input batches and binary target labels.

        Returns
        -------
        probability : numpy.ndarray of shape (len(data_loader.dataset),)
            1D array of predicted probabilities for the positive class.
        y_true : numpy.ndarray of shape (len(data_loader.dataset),)
            1D integer array of true binary labels.
        """

        self.model.eval()  # Validation/Test mode
        probs, labels = [], []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float().unsqueeze(1)
                batch_probs = torch.sigmoid(self.model(batch_X))
                probs.append(batch_probs.cpu().numpy())
                labels.append(batch_y.cpu().numpy())

        probability = np.concatenate(probs).ravel()
        y_true = np.concatenate(labels).ravel().astype(int)
        return probability, y_true

    def accuracy_per_epoch(self, data_loader, threshold=0.5):
        """
        Compute classification accuracy on a DataLoader at a given probability threshold.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing input batches and binary target labels.
        threshold : float, optional
            Probability cutoff above which predictions are assigned to "class 1" (default is 0.5).

        Returns
        -------
        accuracy : float
            Classification accuracy score in the range [0.0, 1.0].
        """

        probability, y_true = self.predict_probabilities(data_loader)
        y_pred = (probability >= threshold).astype(int)
        return float(accuracy_score(y_true, y_pred))

    def train_model(self, config_params=None):
        """
        Train the CNN model across all configured epochs and plot loss and accuracy curves.

        Parameters
        ----------
        config_params : dict, optional
            Hyperparameter dictionary for the network. If None, defaults to 'self.param_configs[0]'.

        Returns
        -------
        train_loss_history : list of floats
            Training loss recorded at each epoch.
        val_loss_history : list of floats
            Validation loss recorded at each epoch.
        train_accuracy_history : list of floats
            Training accuracy recorded at each epoch.
        val_accuracy_history : list of floats
            Validation accuracy recorded at each epoch.
        """

        print("TRAINING")

        # Fall back to the first configuration if None is provided
        if config_params is None:
            config_params = self.param_configs[0]
        
        # Extract network parameters
        batch_size = config_params['batch_size']
        epochs = config_params['epochs']

        # Prepare data to be processed in the loop over epochs
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        for epoch in range(epochs):

            train_loss = self.train_per_epoch(train_loader)
            val_loss = self.validate_per_epoch(validation_loader)
            train_acc = self.accuracy_per_epoch(train_loader)
            val_acc = self.accuracy_per_epoch(validation_loader)

            # Compute validation ROC-AUC for ReduceLROnPlateau
            val_probs, val_true = self.predict_probabilities(validation_loader)
            val_auc = float(roc_auc_score(val_true, val_probs))
            self.scheduler.step(val_auc)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_accuracy_history.append(train_acc)    
            val_accuracy_history.append(val_acc)        
            
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, | Train accuracy = {train_acc:.4f}, Validation accuracy = {val_acc:.4f}")

        print('Finished Training')

        # Plot train and validation loss curves and accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(train_loss_history, label='Train Loss')
        ax1.plot(val_loss_history, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss Over Epochs')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(train_accuracy_history, label='Train Accuracy')
        ax2.plot(val_accuracy_history, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy Over Epochs')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()        

        return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

    def evaluate_optimal_threshold(self, data_loader, threshold=None, verbose=True):
        """
        Evaluate the classification model's performances on a test dataset and dynamically calculates an optimal threshold using 
        Youden's J statistic (derived from the ROC curve) to achieve the best possible balance between True Positive Rate (TPR) and 
        False Positive Rate (FPR).

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing input batches and binary target labels.
        threshold : float, optional
            Probability cutoff above which predictions are assigned to "class 1" (default is 0.5).
        verbose : bool, optional
            Whether to print threshold statistics, ROC-AUC, accuracy, classification report,
            and class distributions (default is True).       

        Returns
        -------
        y_pred : numpy.ndarray of shape (len(data_loader.dataset),)
            Binary class predictions computed using the optimal threshold.
        probability : numpy.ndarray of shape (len(data_loader.dataset),)
            Predicted probabilities for the positive class (Planet).
        active_threshold : float
            The classification threshold applied to produce 'y_pred' (either the provided 'threshold' or the optimal threshold 
            from Youden's J statistic).
        """

        # Compute the model's probability that each test sample is a planet, and get the true label
        probability, y_true = self.predict_probabilities(data_loader)

        # Get False Positive Rates (FPR) and True Positive Rates (TPR) and calculate Youden's J statistic to find the optimal balance
        fpr, tpr, thresholds = roc_curve(y_true, probability)
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])

        # Compare predictions with default and optimal thresholds
        active_threshold = optimal_threshold if threshold is None else float(threshold)
        y_pred = (probability >= active_threshold).astype(int)
        y_pred_default = (probability >= 0.5).astype(int)
        acc_active = float(accuracy_score(y_true, y_pred))
        acc_default = float(accuracy_score(y_true, y_pred_default))
        auc = float(roc_auc_score(y_true, probability))

        # Print the classification report at the optimal threshold (if verbose=True)
        if verbose:
            print("\nTHRESHOLD OPTIMIZATION AND EVALUATION")
            print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f} (default=0.5)")
            print(f"Evaluated threshold: {active_threshold:.4f}")
            print(f"At optimal threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            print(f"Accuracy (default=0.5): {acc_default:.4f} ({acc_default*100:.2f}%)")
            print(f"Accuracy (threshold={active_threshold:.4f}): {acc_active:.4f} ({acc_active*100:.2f}%)")
            print("\nClassification report:")
            print(classification_report(y_true, y_pred, target_names=['Non-Planet', 'Planet'], digits=4, zero_division=0))
            print("Prediction distribution:")
            print(f"  Predicted 0: {(y_pred == 0).sum()}")
            print(f"  Predicted 1: {(y_pred == 1).sum()}")
            print("True distribution:")
            print(f"  True 0: {(y_true == 0).sum()}")
            print(f"  True 1: {(y_true == 1).sum()}")

        return y_pred, probability, active_threshold

    def compute_confusion_matrix(self, y_true, y_pred, threshold, save_path=None):
        """
        Evaluate, print and plot the confusion matrix and eventually save the resulting image.
        
        Parameters 
        ----------
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted binary labels.
        threshold : float
            Classification threshold associated with 'y_pred', used for display titles.
        save_path : str or path-like, optional
            File path to save the confusion matrix plot. If None, the plot is not saved (default None).       

        Returns
        -------
        cm: numpy.ndarray of shape (2,2)
            Confusion matrix computed on y_test using Youden's J threshold.
        """

        # Compute the confusion matrix using the scikit learn confusion_matrix function and print it
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: Non-Planet', 'Actual: Planet'], 
                             columns=['Pred: Non-Planet', 'Pred: Planet']
                             )
        print("\nCONFUSION MATRIX EVALUATION")
        print(f"Confusion Matrix @ threshold {threshold:.3f}")
        print(cm_df)

        # Plot the confusion matrix and eventually save it as a file
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
         for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Confusion matrix plot saved to: {save_path}")  
        plt.show()
        plt.close(fig)
        gc.collect()

        return cm

    def test_model(self, config_params=None):
        """
        Find an optimal threshold on the validation set (via Youden's J) and evaluate the model 
        on the test set applying that threshold, and compute test evaluation metrics and test
        confusion matrix.

        Parameters
        ----------
        config_params : dict, optional
            Hyperparameter dictionary for the network. If None, defaults to 'self.param_configs[0]'.

        Returns
        -------
        cm: numpy.ndarray of shape (2,2)
            Confusion matrix computed on the test using Youden's J threshold found on the validation set.  
        metrics : tuple
            A tuple containing the evaluation metrics (probability, threshold).
        """

        # Fall back to the first configuration if None is provided
        if config_params is None:
            config_params = self.param_configs[0]
                
        # Extract network parameters
        batch_size = config_params['batch_size']

        # Find optimal threshold on validation set
        validation_loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        _, _, optimal_threshold = self.evaluate_optimal_threshold(validation_loader, verbose=False)

        # Apply optimal threshold to the test set
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False)
        prob_test, y_true = self.predict_probabilities(test_loader)
        y_pred_test, prob_test, optimal_threshold = self.evaluate_optimal_threshold(test_loader, threshold=optimal_threshold)
        metrics_test = prob_test, optimal_threshold
        cm = self.compute_confusion_matrix(y_true, y_pred_test, optimal_threshold)

        return cm, metrics_test

    def run_all_configurations(self):
        """
        Iterate through all loaded configurations of hyperparameters using the same data split.
        
        Returns
        -------
        results : list of tuples
            A list where each element contains a tuple of '(params, conf_matrix, metrics)' for each evaluated 
            hyperparameter configuration.
         """
        
        results = []
        for i, params in enumerate(self.param_configs, start=1):
            print(f"\n==============================")
            print(f"Running Configuration {i}: {params}")
            print(f" ")
                    
            # Re-initialize the model with this specific configuration of hyperparameters
            self.model, self.optimizer, self.scheduler, self.loss_function = self.build_cnn(params)
            self.train_model(params)
            conf_matrix, metrics = self.test_model(params)
            results.append((params, conf_matrix, metrics))
    
        return results

if __name__ == "__main__":

    # Import data and files of parameters
    csv_path = Path(__file__).resolve().with_name("tess_data.csv")
    config = {"dataset_path": csv_path}

    # Initialize CNN Classifier and call this method
    cnn_classifier = CNN_classifier(config=config)
    results_all = cnn_classifier.run_all_configurations()