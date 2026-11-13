# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    """...."""

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

        if RF_params is not None:
            default_params.update(RF_params)
            
        self.RF_params = default_params

        # Initialize the scikit-learn Random Forest model
        self.model = RandomForestClassifier(**self.RF_params)

    def load_data(self):
        """Load data already organized properly by calling the class DataSetCreation from dataset.py file"""
        loader = DataSetCreation(csv_path=self.csv_path, mode="machine_learning")
        X_train, X_test, y_train, y_test, meta, X_test_raw, X_err, scaler = loader.load_data()
        print("Funziona!!!")
        return X_train, X_test, y_train, y_test, meta, X_test_raw, X_err, scaler

    def train_model(self, X_train, y_train):
        """Fit the Random Forest classifier on the balanced training set."""
        self.model.fit(X_train, y_train)
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
            print(f"OOB Score: {self.model.oob_score_:.4f}")
        return self

    def evaluate_with_optimal_threshold(self, X_test, y_test):
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
        probability = self.model.predict_proba(X_test)[:, 1]

        # Get False Positive Rates (FPR) and True Positive Rates (TPR) and calculate Youden's J statistic to find the optimal balance
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        # Compare predicions with default and optimal thresholds
        y_pred_default = (probability >= 0.5).astype(int)
        y_pred_best = (probability >= best_thresh).astype(int)
        auc = roc_auc_score(y_test, probability)
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_best = accuracy_score(y_test, y_pred_best)

        # Print the classification report at the optimal threshold
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
    
        print(classification_report(y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
        
        return y_pred_best, probability, best_thresh, (fpr, tpr, thresholds)
