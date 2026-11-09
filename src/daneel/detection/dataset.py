from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset


# Select the mode used when running this module as a script.
DATA_MODE = "machine_learning"  # Set to "deep_learning" to build PyTorch datasets.
RANDOM_STATE = 42  # Shared seed for splitting and balancing.


class DataSetCreation:
    """Prepare TESS light curves for scikit-learn or PyTorch.

    Parameters
    ----------
    csv_path : str or path-like
        Source CSV containing light-curve flux bins, labels, and TIC identifiers.
    mode : {"machine_learning", "deep_learning"}, optional
        Output format selected by prepare(). Default is "machine_learning".
    """

    def __init__(self, csv_path=None, *, mode="machine_learning"):
        """Configure the CSV source and the output mode used by prepare()."""
        if not isinstance(mode, str) or mode not in ("machine_learning", "deep_learning"):
            raise ValueError("mode must be 'machine_learning' or 'deep_learning'.")
        if csv_path is None:
            raise TypeError("csv_path is required.")

        # Keep the legacy model argument for existing callers.
        self.csv_path = csv_path
        self._mode = mode

    def prepare(self, n_bins=1000, use_scaler=True, samples_per_class=350):
        """Prepare datasets for the configured mode.

        Parameters
        ----------
        n_bins : int, optional
            Number of flux bins to read. Default is 1000.
        use_scaler : bool, optional
            Apply per-bin standardization. Default is True in both modes.
        samples_per_class : int, optional
            Number of training samples per class after balancing. Default is 350.

        Returns
        -------
        tuple
            The eight values documented in load_data() for machine learning,
            or the three TensorDataset objects documented in load_pytorch_data().
        """
        deep_learning = self._mode == "deep_learning"
        load = self.load_pytorch_data if deep_learning else self.load_data
        return load(
            n_bins=n_bins,
            use_scaler=use_scaler,
            samples_per_class=samples_per_class,
        )

    @staticmethod
    def _validate_positive_integer(value, name):
        """Reject invalid sizes before indexing or allocating arrays."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _require_columns(df, columns):
        """Report missing CSV fields before selecting columns."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    def create_balanced_dataset(self, X, y, samples_per_class=400, *, random_state=RANDOM_STATE):
        """Balance both classes by subsampling or augmenting their light curves.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_bins)
            Input light curves.
        y : array-like of shape (n_samples,)
            Binary labels, with 0 for negative examples and 1 for planets.
        samples_per_class : int, optional
            Target sample count for each class. Default is 400.
        random_state : int, optional
            Seed for sampling, augmentation, and shuffling. Default is 42.

        Returns
        -------
        X_balanced : numpy.ndarray of shape (2 * samples_per_class, n_bins)
            Original and augmented curves shuffled together.
        y_balanced : numpy.ndarray of shape (2 * samples_per_class,)
            Binary labels aligned with the shuffled curves.

        Notes
        -----
        Augmentation chooses one of four operations with equal probability:
        Gaussian noise (standard deviation 0.01), scaling (0.97 to 1.03),
        a circular shift (-20 to 19 bins), or scaling (0.98 to 1.02) with
        Gaussian noise (standard deviation 0.008).
        """
        self._validate_positive_integer(samples_per_class, "samples_per_class")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be a 2-D array and y must be a matching 1-D array.")

        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        if len(X_class0) == 0 or len(X_class1) == 0 or len(X_class0) + len(X_class1) != len(y):
            raise ValueError("Training labels must include both classes 0 and 1 and no other labels.")

        # Use a local random generator for reproducible balancing.
        rng = np.random.RandomState(random_state)
        print(f"Train before balancing - Class 0: {len(X_class0)}, Class 1: {len(X_class1)}")

        def augment_to_target(X_orig, n_target):
            """Downsample a class or generate the missing curves."""
            if len(X_orig) >= n_target:
                idx = rng.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]

            # Keep the original curves and sample from them for augmentation.
            X_result = [X_orig]
            n_needed = n_target - len(X_orig)
            while n_needed > 0:
                idx = rng.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = rng.rand()
                if aug_type < 0.25:
                    # Add Gaussian noise to the flux.
                    X_aug = X_orig[idx] + rng.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    # Apply a small scaling factor to each curve.
                    scale = 1.0 + rng.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    # Shift each curve circularly along the time bins.
                    shifts = rng.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    # Combine a small scaling change with Gaussian noise.
                    X_aug = X_orig[idx] * (1.0 + rng.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += rng.normal(0, 0.008, X_aug.shape)

                X_result.append(X_aug)
                n_needed -= len(X_aug)

            return np.vstack(X_result)

        X0_bal = augment_to_target(X_class0, samples_per_class)
        X1_bal = augment_to_target(X_class1, samples_per_class)
        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")

        X_balanced = np.vstack([X0_bal, X1_bal])
        y_balanced = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

        # Shuffle curves and labels with the same permutation.
        idx = np.arange(len(X_balanced))
        rng.shuffle(idx)
        return X_balanced[idx], y_balanced[idx]

    def _read_csv(self, n_bins):
        """Read the CSV and extract flux bins and binary labels."""
        self._validate_positive_integer(n_bins, "n_bins")
        flux_cols = [f"flux_{i:04d}" for i in range(n_bins)]
        df = pd.read_csv(self.csv_path)
        self._require_columns(df, ["label", "tic", *flux_cols])
        X = df[flux_cols].to_numpy()
        y = df["label"].to_numpy()

        n_class0 = (y == 0).sum()
        n_class1 = (y == 1).sum()
        if n_class0 + n_class1 != len(y):
            raise ValueError("Dataset labels must be 0 or 1.")

        print(f"Dataset: {len(df)} samples")
        print(f"Full CSV - Class 0: {n_class0}, Class 1: {n_class1}")
        return df, X, y

    @staticmethod
    def _split_indices(y, tics, n_folds=5):
        """Split row indices, reserving roughly 1/n_folds for the holdout.

        Stratify by binary label and keep repeated TIC identifiers together
        so observations of the same star stay on one side of the split.
        """
        indices = np.arange(len(y))
        tic_groups, known_tics = pd.factorize(tics)

        # Give each row without a TIC identifier its own group.
        missing_tic = tic_groups == -1
        tic_groups[missing_tic] = len(known_tics) + indices[missing_tic]
        n_groups = len(known_tics) + missing_tic.sum()

        # Use label stratification alone when each TIC has one row.
        if n_groups == len(y):
            return train_test_split(
                indices, test_size=1 / n_folds, random_state=RANDOM_STATE, stratify=y
            )

        # Find a fold with both classes on each side and disjoint TIC groups.
        n_splits = min(n_folds, n_groups, (y == 0).sum(), (y == 1).sum())
        if n_splits < 2:
            raise ValueError("Need at least two TIC groups and two samples per class to split the dataset.")

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for idx_train, idx_holdout in splitter.split(indices, y, groups=tic_groups):
            if len(np.unique(y[idx_train])) == len(np.unique(y[idx_holdout])) == 2:
                return idx_train, idx_holdout

        raise ValueError("Could not create a TIC-grouped split with both classes in each part.")

    def _prepare_splits(self, X, y, indices, *, use_scaler, samples_per_class):
        """Return (flux, label) pairs in input split order and the fitted scaler.

        Balance the first split and fit the scaler on it when requested.
        Apply the same scaler to any evaluation splits.
        """
        idx_train, *idx_eval = indices
        X_train, y_train = self.create_balanced_dataset(
            X[idx_train], y[idx_train], samples_per_class=samples_per_class
        )

        # Fit the scaler on the balanced training data.
        scaler = None
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        splits = [(X_train, y_train)]
        for idx in idx_eval:
            X_eval = X[idx]
            if scaler is not None:
                X_eval = scaler.transform(X_eval)
            splits.append((X_eval, y[idx]))
        return splits, scaler

    def load_data(self, n_bins=1000, use_scaler=True, samples_per_class=350):
        """Load and preprocess TESS light curves for machine learning.

        Parameters
        ----------
        n_bins : int, optional
            Number of flux bins to read. Default is 1000.
        use_scaler : bool, optional
            Standardize each bin using the training statistics. Default is True.
        samples_per_class : int, optional
            Number of training samples per class after balancing. Default is 350.

        Returns
        -------
        X_train : numpy.ndarray of shape (2 * samples_per_class, n_bins)
            Balanced training fluxes, standardized if requested.
        X_test : numpy.ndarray of shape (n_test_samples, n_bins)
            Test fluxes on the same scale as X_train, ready for prediction.
        y_train : numpy.ndarray of shape (2 * samples_per_class,)
            Binary labels for the balanced training set.
        y_test : numpy.ndarray of shape (n_test_samples,)
            Binary labels for the test set.
        metadata_test : pandas.DataFrame
            Test rows' toi_name, tic, label, disp, period_d, t0_bjd, dur_hr, and sector.
        X_test_raw : numpy.ndarray of shape (n_test_samples, n_bins)
            Original CSV fluxes for plotting directly with X_err_test.
        X_err_test : numpy.ndarray of shape (n_test_samples, n_bins)
            Flux uncertainties in the original CSV units, aligned with X_test_raw.
        scaler : StandardScaler or None
            Scaler fitted on the balanced training data, or None if disabled.

        Notes
        -----
        The split targets an 80/20 train/test ratio and keeps repeated TICs
        together. Test curves retain their original sample distribution.

        Examples
        --------
        .. code-block:: python

            loader = DataSetCreation(csv_path="tess_data.csv")
            X_train, X_test, y_train, y_test, meta, X_test_raw, X_err, scaler = loader.load_data()
            # With the defaults, X_train.shape is (700, 1000).

        Use X_test for model predictions and X_test_raw with X_err to plot a
        test light curve in the original flux units.
        """
        df, X, y = self._read_csv(n_bins)
        metadata_cols = ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
        flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins)]
        self._require_columns(df, metadata_cols + flux_err_cols)

        idx_train, idx_test = self._split_indices(y, df["tic"])
        # NumPy indexing by row indices makes an independent, unscaled array.
        X_test_raw = X[idx_test]
        (train, test), scaler = self._prepare_splits(
            X, y, (idx_train, idx_test), use_scaler=use_scaler, samples_per_class=samples_per_class
        )
        X_train, y_train = train
        X_test, y_test = test

        # Keep metadata and flux uncertainties aligned with the test rows.
        test_rows = df.iloc[idx_test]
        metadata_test = test_rows[metadata_cols].reset_index(drop=True)
        X_err_test = test_rows[flux_err_cols].to_numpy()

        print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, metadata_test, X_test_raw, X_err_test, scaler

    def load_pytorch_data(self, n_bins=1000, use_scaler=True, samples_per_class=350):
        """Load and preprocess TESS light curves for PyTorch.

        Parameters
        ----------
        n_bins : int, optional
            Number of flux bins per curve. Default is 1000.
        use_scaler : bool, optional
            Standardize each bin using the training statistics. Default is True.
        samples_per_class : int, optional
            Number of training samples per class after balancing. Default is 350.

        Returns
        -------
        train_dataset, test_dataset, val_dataset : torch.utils.data.TensorDataset
            Datasets in train, test, validation order. Each example contains
            float32 fluxes of shape (1, n_bins) for Conv1d and a scalar float32
            label for BCEWithLogitsLoss.

        Notes
        -----
        The split targets a 60/20/20 ratio before balancing. Balancing and
        fitting the scaler use only the training split.

        Examples
        --------
        .. code-block:: python

            loader = DataSetCreation(csv_path="tess_data.csv", mode="deep_learning")
            train, test, validation = loader.prepare()
            # With the defaults, each flux tensor has shape (1, 1000).
        """
        df, X, y = self._read_csv(n_bins)
        idx_pool, idx_test = self._split_indices(y, df["tic"])

        # Reserve about one quarter of the remaining rows for validation.
        # This yields an approximate 60/20/20 split before balancing.
        train_rel, val_rel = self._split_indices(y[idx_pool], df["tic"].iloc[idx_pool], n_folds=4)
        idx_train, idx_val = idx_pool[train_rel], idx_pool[val_rel]
        splits, _ = self._prepare_splits(
            X, y, (idx_train, idx_test, idx_val), use_scaler=use_scaler, samples_per_class=samples_per_class
        )

        # Place the channel dimension before the time bins for Conv1d.
        train_dataset, test_dataset, val_dataset = (
            TensorDataset(
                torch.from_numpy(features.astype(np.float32, copy=False)).unsqueeze(1),
                torch.from_numpy(labels.astype(np.float32, copy=False)),
            )
            for features, labels in splits
        )
        print(f"Final - Train: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(val_dataset)}")
        return train_dataset, test_dataset, val_dataset


if __name__ == "__main__":
    # Locate the example CSV next to this module.
    csv_path = Path(__file__).resolve().with_name("tess_data.csv")
    DataSetCreation(csv_path=csv_path, mode="deep_learning").prepare()
