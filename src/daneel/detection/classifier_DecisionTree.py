# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve
)


from .dataset import DataSetCreation

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Definition of a class implementing Decision Tree classifier
class DecisionTree_classifier:
    """Class implementing Decision Tree classifier supporting single or multiple hyperparameter sets."""

    def __init__(self, config):
        """
        Initialize the Decision Tree classifier configuration and load the dataset.
    
        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from the YAML parameters file, containing:
            - dataset_path : str or path-like
                Path to the input CSV file containing the data. Required.
            - DT_params : dict or list of dict, optional
                Hyperparameters for the Decision Tree classifier. If a dictionary is provided, 
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

        # Define default hyperparameters and update them if other input hyperparameters are given
        default_params = {
            #'splitter': best,
            'max_depth': 15,
            'min_samples_split': 2,
            'max_features': 'sqrt', 
            'class_weight': None,
            'random_state': RANDOM_STATE
            }

        # Import hyperparameters of the model from the configuration file
        raw_params = config.get("DT_params")
        self.param_configs = []

        # Handle whether DT_params is a single dict, a list of dicts, or None
        if raw_params is None or raw_params == {}:  # No hyperparameters in input
            self.param_configs.append(default_params)
        elif isinstance(raw_params, list):  # List of different sets of hyperparameters
            self.param_configs = []
            for params in raw_params:
                p = default_params.copy()
                p.update(params)
                self.param_configs.append(p)
        else:
            default_params = default_params.copy()  # Single set of parameters (or None)
            if raw_params is not None:
                default_params.update(raw_params)
            self.DT_params = default_params
            self.param_configs = [default_params]

        # Initialize model with the first configuration by default
        self.model = DecisionTreeClassifier(**self.param_configs[0])
        self.load_data()

    def load_data(self):
        """Load data properly formatted by calling the DataSetCreation class from dataset.py in "machine_learning" mode."""
        loader = DataSetCreation(csv_path=self.csv_path, mode="machine_learning")
        self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler = loader.load_data()
        
        # Extract names of the features, to use in the plot_decision_tree function
        self.feature_names = [f"flux_{i:04d}" for i in range(self.X_train.shape[1])]

        return self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler

    def train_model(self):
        """Fit the Decision Tree classifier on the balanced training set."""
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_with_optimal_threshold(self, save_model=False, model_prefix="tess_dt_model"):
    
        """
        Evaluate the classification model's performances on a test dataset and dynamically calculates an optimal threshold using 
        Youden’s J statistic (derived from the ROC curve) to achieve the best possible balance between True Positive Rate (TPR) and 
        False Positive Rate (FPR), and optionally saves the model and threshold.

        Parameters
        ----------
        save_model : bool, optional
            If True, saves the trained model and optimal threshold to disk (default is False).
        model_prefix : str, optional
            Prefix for the saved model and threshold files (default is "tess_dt_model").

        Returns
        -------
        y_pred_best : numpy.ndarray of shape (n_test_samples,)
            Binary class predictions computed using the optimal threshold.
        probability : numpy.ndarray of shape (n_test_samples,)
            Predicted probabilities for the positive class (Planet).
        best_thresh : float
            The optimal classification threshold determined via Youden's J statistic.
        """

        # Compute the model's probability that each test sample is a planet
        probability = self.model.predict_proba(self.X_test)[:, 1]

        # Get False Positive Rates (FPR) and True Positive Rates (TPR) and calculate Youden's J statistic to find the optimal balance
        fpr, tpr, thresholds = roc_curve(self.y_test, probability)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        # Compare predicions with default and optimal thresholds
        y_pred_default = (probability >= 0.5).astype(int)
        y_pred_best = (probability >= best_thresh).astype(int)
        auc = roc_auc_score(self.y_test, probability)
        acc_default = accuracy_score(self.y_test, y_pred_default)
        acc_best = accuracy_score(self.y_test, y_pred_best)

        # Print the classification report at the optimal threshold
        print("THRESHOLD OPTIMIZATION AND EVALUATION")
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
    
        print(classification_report(self.y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))

        # Save trained model and optimal classification threshold to disk
        if save_model is True:
            model_filename = f"{model_prefix}.joblib"
            thresh_filename = f"{model_prefix}_optimal_threshold.npy"
            
            joblib.dump(self.model, model_filename)
            np.save(thresh_filename, best_thresh)
            print(f"Saved model: {model_filename}, {thresh_filename}")
        
        return y_pred_best, probability, best_thresh

    def compute_confusion_matrix(self, save_path=None):
        """Evaluate, print and plot the confusion matrix and eventually save the resulting image.
        
        Parameters 
        ----------
        save_path : str or path-like, optional
            File path to save the confusion matrix plot. If None, the plot is not saved (default None).

        Returns
        -------
        cm: numpy.ndarray of shape (2,2)
            Confusion matrix with y_true as rows and y_pred as columns.  
        metrics : tuple
            A tuple containing the evaluation metrics (probability, threshold).
        """
        print(" ")

        # Defining the input parameters
        y_true = self.y_test
        y_pred, probability, threshold = self.evaluate_with_optimal_threshold()
        metrics = probability, threshold

        # Compute the confusion matrix using the scikit learn confusion_matrix function and print it
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: Non-Planet', 'Actual: Planet'], 
                             columns=['Pred: Non-Planet', 'Pred: Planet']
                             )
        print("CONFUSION MATRIX EVALUATION")
        print(f"Confusion Matrix @ threshold {threshold:.3f})")
        print(cm_df)

        # Plot the confusion matrix and eventually save it save as a .png file
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

        return cm, metrics

    def plot_decision_tree(self):


        plt.figure(figsize=(40, 20), dpi=600)

        plot_tree(
        self.model, 
        feature_names=self.feature_names, 
        class_names=['Non-Planet', 'Planet'],
        max_depth=3, 
        filled=True, 
        fontsize=6
        )

        plt.savefig('decision_tree.pdf', bbox_inches='tight')
        plt.show()

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
            self.model = DecisionTreeClassifier(**params)
            self.train_model()
            conf_matrix, metrics = self.compute_confusion_matrix()
            results.append((params, conf_matrix, metrics))

        return results


#if __name__ == "__main__":

    # Import data and files of parameters
#    csv_path = Path(__file__).resolve().with_name("tess_data.csv")

    # Initialize Decision Tree classifier and call this method
 #   dt_classifier = DecisionTree_classifier(csv_path=csv_path)
  #  results_all = dt_classifier.run_all_configurations()


if __name__ == "__main__":

    config = {
        "dataset_path": str(
            Path(__file__).resolve().with_name("tess_data.csv")
        ),

        "DT_params": {
            "max_depth": 10,
            "min_samples_leaf": 5,
            "max_features": None
        }
    }

    dt_classifier = DecisionTree_classifier(config)

    dt_classifier.train_model()

    dt_classifier.compute_confusion_matrix()

    dt_classifier.plot_decision_tree()