# Imports and configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve
)


from dataset import DataSetCreation

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Definition of a class implementing Random Forest classifier
class RandomForest_classifier:
    """Class implementing Random Forest classifier supporting single or multiple hyperparameter sets."""

    def __init__(self, csv_path=None, RF_params=None):
        """
        Initialize the Random Forest classifier configuration and load the dataset.

        Parameters
        ----------
        csv_path : str or path-like
            Path to the input CSV file containing the data. Required.
        RF_params : dict or list of dict, optional
            Hyperparameters for the Random Forest classifier. If a dictionary is provided, 
            it contains a single configuration. If a list of dictionaries is provided, 
            multiple configurations will be evaluated (default is None, which uses default 
            hyperparameters).
        """

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
        """Load data properly formatted by calling the DataSetCreation class from dataset.py in "machine_learning" mode."""
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

        # Compute the confusion matrix using the scikit learn confusion_matrix function
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: Non-Planet', 'Actual: Planet'], 
                             columns=['Pred: Non-Planet', 'Pred: Planet']
                             )

        # Print the confusion matrix
        tn, fp, fn, tp = cm.ravel()
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
            self.model = RandomForestClassifier(**params)
            self.train_model()
            conf_matrix, metrics = self.compute_confusion_matrix()
            results.append((params, conf_matrix, metrics))

        return results


if __name__ == "__main__":

    # Import data and files of parameters
    csv_path = Path(__file__).resolve().with_name("tess_data.csv")

    # Initialize Random Forest Classifier and call this method
    rf_classifier = RandomForest_classifier(csv_path=csv_path)
    results_all = rf_classifier.run_all_configurations()