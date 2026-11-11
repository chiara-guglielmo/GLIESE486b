# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

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

        # Load data already organized
        loader = DataSetCreation(csv_path="tess_data.csv", mode="machine_learning")
        self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler = loader.load_data()

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


def train_model(self):
    """..."""
    self.model.fit(self.X_train, self.y_train)
    if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
        print(f"OOB Score: {self.model.oob_score_:.4f}")
    return self

def fit(self, X, y):
    raise NotImplementedError("Random-forest training is a course exercise")