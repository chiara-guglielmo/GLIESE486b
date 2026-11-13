# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)


from dataset import DataSetCreation

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Definition of a class implementing Random Forest classifier
class RandomForest_classifier:
    """Class implementing Random Forest classifier supporting single or multiple hyperparameter sets."""

    def __init__(self, csv_path=None, RF_params=None):

        if csv_path is None:
            raise TypeError("csv_path is required.")
    
        self.csv_path = csv_path

        # Define default hyperparameters and update them if other input hyperparameters are given
        default_params = {
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': None,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'oob_score': True
        }

        # Handle whether RF_params is a single dict, a list of dicts, or None
        if isinstance(RF_params, list):  # List of different sets of hyperparameters
            self.param_configs = []
            for params in RF_params:
                p = default_params.copy()
                p.update(params)
                self.param_configs.append(p)
            self.model = RandomForestClassifier(**self.param_configs[0])
        else:
            default_params = default_params.copy()  # Single set of parameters (or None)
            if RF_params is not None:
                default_params.update(RF_params)
            self.RF_params = default_params
            self.param_configs = [default_params]
            self.model = RandomForestClassifier(**self.RF_params)

        self.load_data()

    def load_data(self):
        """Load data already organized properly by calling the class DataSetCreation from dataset.py file"""
        loader = DataSetCreation(csv_path=self.csv_path, mode="machine_learning")
        self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler = loader.load_data()
        return self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler

    def train_model(self):
        """Fit the Random Forest classifier on the balanced training set."""
        self.model.fit(self.X_train, self.y_train)
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
            print(f"OOB Score: {self.model.oob_score_:.4f}")
        return self

    def evaluate_with_optimal_threshold(self):
        """
        Evaluate the classification model's performances on a test dataset and dynamically calculates an optimal threshold using 
        Youden’s J statistic (derived from the ROC curve) to achieve the best possible balance between True Positive Rate (TPR) and 
        False Positive Rate (FPR).

        Parameters 
        ----------
        X_test, y_test
        """
        print(" ")

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
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
    
        print(classification_report(self.y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
        
        return y_pred_best, probability, best_thresh, (fpr, tpr, thresholds)

    def run_all_configurations(self):
        """Iterate through all loaded configurations of hyperparameters using the same data split."""
        results = []
        for i, params in enumerate(self.param_configs, start=1):
            print(f"\n==============================")
            print(f"Running Configuration {i}: {params}")
            print(f"==============================")
            
            # Re-initialize the model with this specific configuration of hyperparameters
            self.model = RandomForestClassifier(**params)
            self.train_model()
            metrics = self.evaluate_with_optimal_threshold()

            results.append((params, metrics))
        return results


if __name__ == "__main__":

    # Import data and files of parameters
    csv_path = Path(__file__).resolve().with_name("tess_data.csv")

    # Initialize Random Forest Classifier and call this method
    rf_classifier = RandomForest_classifier(csv_path=csv_path)
    results_all = rf_classifier.run_all_configurations()