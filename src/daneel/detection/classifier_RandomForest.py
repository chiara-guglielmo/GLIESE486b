# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import DataSetCreation
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class RFClassifier:

    """...."""

    # Defining the Random Forest model and hyperparameters

    def __init__(self, RF_params=None):

        # default parameters
        RF_default_params = {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': None,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'oob_score': True}

        # adjusting to default parameters or updating them
        if RF_params is None:
            RF_params = RF_default_params
        else:
            RF_default_params.update(RF_params)
            RF_params = RF_default_params

        # memorizing (updated or default) parameters
        self.RF_params = RF_params

        # defining the RF model as the RandomForestClassifier from scikit-learn
        RF_model = RandomForestClassifier(**RF_params)
        self.model = RF_model

    # Training the Random Forest model

    def train_RF(self, csv_path=None):

            # adjusting to a default CSV or updating the path 
            default_csv = 'tess_data.csv'  
            
            if csv_path is None:
                csv_path = default_csv

            # accessing data from the existing dataset
            loader = DataSetCreation(csv_path, mode='machine_learning')
            self.X_train, self.X_test, self.y_train, self.y_test, self.meta, self.X_test_raw, self.X_err, self.scaler = loader.load_data()

            # training
            self.model.fit(self.X_train, self.y_train)
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                print(f"OOB Score: {self.model.oob_score_:.4f}")
            
            return self

    # Evaluating the model

    