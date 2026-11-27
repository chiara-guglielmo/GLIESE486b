
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Generator(nn.Module):

    def __init__(self, input_size=100, ngf=64):
        super().__init__()

        self.input_size = input_size

        self.conv_model = nn.Sequential(
            # 1x1 -> 3x3
            nn.ConvTranspose2d(
                input_size, ngf * 8, 3, 1, 0, bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 3x3 -> 5x5
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4, 3, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 5x5 -> 11x11
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, 3, 2, 0, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 11x11 -> 22x22
            nn.ConvTranspose2d(
                ngf * 2, ngf, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 22x22 -> 44x44
            nn.ConvTranspose2d(
                ngf, 1, 4, 2, 1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):

        x = x.reshape(-1, self.input_size, 1, 1)

        return self.conv_model(x)


class Discriminator(nn.Module):

    def __init__(self, ndf=32):
        super().__init__()

        self.model = nn.Sequential(
            # 44x44 -> 22x22
            nn.Conv2d(
                1, ndf, 4, 2, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 22x22 -> 11x11
            nn.Conv2d(
                ndf, ndf * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 11x11 -> 5x5
            nn.Conv2d(
                ndf * 2, ndf * 4, 3, 2, 0, bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 5x5 -> 3x3
            nn.Conv2d(
                ndf * 4, ndf * 8, 3, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 3x3 -> 1x1
            nn.Conv2d(
                ndf * 8, 1, 3, 1, 0, bias=False
            )
        )

    def forward(self, x):

        return self.model(x).reshape(-1)


class GAN:

    def __init__(self, config):

        self.config = config

        # ========================================
        # Dataset path
        # ========================================

        dataset_path = config.get("dataset_path")

        if not dataset_path:
            raise ValueError(
                "dataset_path must be specified "
                "in the YAML configuration."
            )

        self.csv_path = (
            Path(dataset_path).expanduser().resolve()
        )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset CSV file not found: {self.csv_path}"
            )

        # ========================================
        # Default hyperparameters
        # ========================================

        default_params = {
            "batch_size": 32,
            "learning_rate": 2e-4,
            "num_epochs": 500,
            "noise_dim": 100,
            "features_generator": 64,
            "features_discriminator": 32,
            "beta1": 0.5,
            "target_bins": 1000
        }

        raw_params = config.get("GAN_params")

        if raw_params is not None:

            if not isinstance(raw_params, dict):
                raise ValueError(
                    "GAN_params must be a dictionary."
                )

            unknown_params = (
                set(raw_params) - set(default_params)
            )

            if unknown_params:
                raise ValueError(
                    f"Unknown GAN parameters: "
                    f"{sorted(unknown_params)}"
                )

            default_params.update(raw_params)

        self.GAN_params = default_params

        self.batch_size = default_params["batch_size"]

        self.learning_rate = float(
            default_params["learning_rate"]
        )

        self.num_epochs = default_params["num_epochs"]

        self.noise_dim = default_params["noise_dim"]

        self.features_generator = default_params[
            "features_generator"
        ]

        self.features_discriminator = default_params[
            "features_discriminator"
        ]

        self.beta1 = float(default_params["beta1"])

        self.target_bins = default_params["target_bins"]

        self.n_generated_curves = config.get(
            "N_lightcurve", 100
        )

        # ========================================
        # Validate hyperparameters
        # ========================================

        integer_params = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "noise_dim": self.noise_dim,
            "features_generator": self.features_generator,
            "features_discriminator": self.features_discriminator,
            "target_bins": self.target_bins,
            "N_lightcurve": self.n_generated_curves
        }

        for name, value in integer_params.items():

            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if not 0 <= self.beta1 < 1:
            raise ValueError(
                "beta1 must be in [0, 1)."
            )

        # ========================================
        # Device selection
        # ========================================

        device_config = config.get("device")

        if device_config is not None:
            self.device = torch.device(device_config)

        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")

        else:
            self.device = torch.device("cpu")

        print(f"Computer device used: {self.device}")

        # ========================================
        # Model attributes
        # ========================================

        self.generator = None
        self.discriminator = None

        self.gen_opt = None
        self.dis_opt = None

        self.normalization = None

        self.trained = False

        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {
            "generator": [],
            "discriminator": []
        }


    @staticmethod
    def weights_init(m):

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):

            nn.init.normal_(m.weight, 0.0, 0.02)

        elif isinstance(m, nn.BatchNorm2d):

            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


    def build_gan(self):

        self.generator = Generator(
            input_size=self.noise_dim,
            ngf=self.features_generator
        ).to(self.device)

        self.discriminator = Discriminator(
            ndf=self.features_discriminator
        ).to(self.device)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.gen_opt = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, 0.999)
        )

        self.dis_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate / 2,
            betas=(self.beta1, 0.999)
        )


    @staticmethod
    def normalize_images(images):

        # Separate flux and uncertainty
        values = images.reshape(-1, 2, 968)

        flux = values[:, 0]
        error = values[:, 1]

        # Robust scaling parameters
        center = flux.median()

        flux_scale = (
            (flux - center).abs().median().clamp_min(1e-6)
        )

        error_scale = error.median().clamp_min(1e-12)

        # Compress extreme values
        transformed = torch.stack((
            torch.asinh((flux - center) / flux_scale),
            torch.log1p(error / error_scale)
        ), dim=1)

        # Min-Max normalization
        minimum = transformed.amin(
            dim=(0, 2),
            keepdim=True
        )

        scale = (
            transformed.amax(
                dim=(0, 2),
                keepdim=True
            ) - minimum
        )

        divisor = torch.where(
            scale > 0,
            scale,
            torch.ones_like(scale)
        )

        # Map values to [-1, 1]
        normalized = (
            2 * (transformed - minimum) / divisor - 1
        )

        normalization = {
            "center": center,
            "flux_scale": flux_scale,
            "error_scale": error_scale,
            "minimum": minimum,
            "scale": scale
        }

        return normalized.reshape_as(images), normalization


    @staticmethod
    def restore_images(images, normalization):

        values = images.reshape(-1, 2, 968)

        params = {
            key: value.to(images.device)
            for key, value in normalization.items()
        }

        # Inverse Min-Max normalization
        transformed = (
            (values + 1) * params["scale"] / 2
            + params["minimum"]
        )

        # Restore original flux
        flux = (
            torch.sinh(transformed[:, 0])
            * params["flux_scale"]
            + params["center"]
        )

        # Restore original uncertainties
        error = (
            torch.expm1(transformed[:, 1])
            * params["error_scale"]
        )

        return torch.stack(
            (flux, error), dim=1
        ).reshape_as(images)


    def load_data(self):

        # Load original CSV
        df = pd.read_csv(self.csv_path)

        if "label" not in df.columns:
            raise ValueError(
                "The dataset must contain a label column."
            )

        # Select only planetary light curves
        df = df.loc[
            df["label"] == 1
        ].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No label-1 curves found in the dataset."
            )

        # Select flux and uncertainty columns
        flux_cols = [
            f"flux_{i:04d}" for i in range(1000)
        ]

        flux_err_cols = [
            f"flux_err_{i:04d}" for i in range(1000)
        ]

        # Check required columns
        missing_cols = [
            col for col in flux_cols + flux_err_cols
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing dataset columns: "
                f"{missing_cols[:10]}"
            )

        # Convert to NumPy arrays
        flux = df[flux_cols].to_numpy(
            dtype=np.float32
        )

        flux_err = df[flux_err_cols].to_numpy(
            dtype=np.float32
        )

        # Check data validity
        if not np.isfinite(flux).all():
            raise ValueError(
                "Invalid flux values."
            )

        if not np.isfinite(flux_err).all():
            raise ValueError(
                "Invalid flux uncertainties."
            )

        if (flux_err < 0).any():
            raise ValueError(
                "Negative flux uncertainties."
            )

        # Resample from 1000 to 968 bins
        old_grid = np.linspace(0, 1, 1000)
        new_grid = np.linspace(0, 1, 968)

        flux = np.array([
            np.interp(new_grid, old_grid, curve)
            for curve in flux
        ], dtype=np.float32)

        flux_err = np.array([
            np.interp(new_grid, old_grid, curve)
            for curve in flux_err
        ], dtype=np.float32)

        # Combine flux and uncertainty
        real_images = np.stack(
            (flux, flux_err),
            axis=1
        )

        # Convert to PyTorch tensor
        real_images = torch.from_numpy(real_images)

        # Reshape to (N, 1, 44, 44)
        real_images = real_images.reshape(
            -1, 1, 44, 44
        )

        # Normalize the entire dataset
        train, self.normalization = (
            self.normalize_images(real_images)
        )

        print(f"Training samples: {len(train)}")

        print(f"Training shape: {tuple(train.shape)}")

        print(
            f"Training range: "
            f"[{train.min():.6f}, {train.max():.6f}]"
        )

        return train


    def train_step(self, images):

        current_batch_size = images.size(0)

        # Generate random latent vectors
        noise = torch.randn(
            current_batch_size,
            self.noise_dim,
            1,
            1,
            device=self.device
        )

        # Generate fake images
        generated_images = self.generator(noise)

        # ========================================
        # Train Discriminator
        # ========================================

        self.discriminator.train()

        self.dis_opt.zero_grad(set_to_none=True)

        # Real and fake share BatchNorm statistics
        combined_images = torch.cat(
            (images, generated_images.detach()),
            dim=0
        )

        combined_output = self.discriminator(
            combined_images
        )

        real_output = combined_output[
            :current_batch_size
        ]

        fake_output = combined_output[
            current_batch_size:
        ]

        # Discriminator losses
        real_loss = self.criterion(
            real_output,
            torch.ones_like(real_output)
        )

        fake_loss = self.criterion(
            fake_output,
            torch.zeros_like(fake_output)
        )

        disc_loss = 0.5 * (
            real_loss + fake_loss
        )

        disc_loss.backward()
        self.dis_opt.step()

        # ========================================
        # Train Generator
        # ========================================

        self.gen_opt.zero_grad(set_to_none=True)

        # Freeze discriminator weights
        self.discriminator.requires_grad_(False)

        # Keep BatchNorm in training mode
        self.discriminator.train()

        # Freeze BatchNorm running statistics
        bn_layers = [
            layer
            for layer in self.discriminator.modules()
            if isinstance(layer, nn.BatchNorm2d)
        ]

        original_tracking = [
            layer.track_running_stats
            for layer in bn_layers
        ]

        for layer in bn_layers:
            layer.track_running_stats = False

        try:

            # Shared BatchNorm statistics
            combined_images = torch.cat(
                (images, generated_images),
                dim=0
            )

            combined_output = self.discriminator(
                combined_images
            )

            fake_output = combined_output[
                current_batch_size:
            ]

            # Generator adversarial loss
            gen_loss = self.criterion(
                fake_output,
                torch.ones_like(fake_output)
            )

            gen_loss.backward()
            self.gen_opt.step()

        finally:

            # Restore BatchNorm settings
            for layer, tracking in zip(
                bn_layers,
                original_tracking
            ):
                layer.track_running_stats = tracking

            self.discriminator.requires_grad_(True)

        return gen_loss.item(), disc_loss.item()


    def train_model(self):

        torch.manual_seed(42)

        self.trained = False

        # Load entire training dataset
        train = self.load_data()

        # Initialize models
        self.build_gan()

        dataloader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.history = {
            "generator": [],
            "discriminator": []
        }

        start = time.time()

        # ========================================
        # Training loop
        # ========================================

        for epoch in range(self.num_epochs):

            self.generator.train()
            self.discriminator.train()

            total_gen_loss = 0.0
            total_disc_loss = 0.0

            n_samples = 0

            for batch in dataloader:

                batch = batch.to(self.device)

                gen_loss, disc_loss = self.train_step(
                    batch
                )

                if not np.isfinite([gen_loss, disc_loss]).all():
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch + 1}."
                    )

                total_gen_loss += (
                    gen_loss * len(batch)
                )

                total_disc_loss += (
                    disc_loss * len(batch)
                )

                n_samples += len(batch)

            # Average losses over all training samples
            avg_gen_loss = total_gen_loss / n_samples

            avg_disc_loss = total_disc_loss / n_samples

            self.history["generator"].append(
                avg_gen_loss
            )

            self.history["discriminator"].append(
                avg_disc_loss
            )

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}] "
                f"G: {avg_gen_loss:.4f} "
                f"D: {avg_disc_loss:.4f} "
                f"Time: {time.time() - start:.1f}s"
            )

        # Use models from the final epoch
        self.generator.eval()
        self.discriminator.eval()

        self.trained = True

        print("\nGAN training completed.")

        return self.history


    def plot_losses(self):

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        epochs = np.arange(
            1,
            len(self.history["generator"]) + 1
        )

        ax.plot(
            epochs,
            self.history["generator"],
            label="Generator"
        )

        ax.plot(
            epochs,
            self.history["discriminator"],
            label="Discriminator"
        )

        ax.set_title("GAN training losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.legend()
        ax.grid()

        fig.tight_layout()

        return fig


    @torch.no_grad()
    def generate_csv(self, save_path=None):

        if not self.trained:
            raise RuntimeError(
                "The GAN must be trained before generating curves."
            )

        self.generator.eval()

        n_curves = self.n_generated_curves
        n_bins = self.target_bins

        # Generate random latent vectors
        noise = torch.randn(
            n_curves,
            self.noise_dim,
            1,
            1,
            device=self.device
        )

        # Generate synthetic images in batches
        generated_images = torch.cat([
            self.generator(batch).cpu()
            for batch in noise.split(self.batch_size)
        ], dim=0)

        # Restore original flux and uncertainty values
        generated_images = self.restore_images(
            generated_images,
            self.normalization
        )

        # Separate flux and uncertainty
        generated_signals = generated_images.reshape(
            n_curves, 2, 968
        ).numpy()

        flux = generated_signals[:, 0, :]
        flux_err = generated_signals[:, 1, :]

        # Resample to the requested number of bins
        original_bins = flux.shape[1]

        if original_bins != n_bins:

            old_grid = np.linspace(
                0, 1, original_bins
            )

            new_grid = np.linspace(
                0, 1, n_bins
            )

            flux = np.array([
                np.interp(new_grid, old_grid, curve)
                for curve in flux
            ])

            flux_err = np.array([
                np.interp(new_grid, old_grid, curve)
                for curve in flux_err
            ])

        # Validate generated values
        if not np.isfinite(flux).all():
            raise ValueError(
                "Generated flux contains invalid values."
            )

        if not np.isfinite(flux_err).all():
            raise ValueError(
                "Generated uncertainties contain invalid values."
            )

        # Create metadata
        metadata = pd.DataFrame({
            "toi_name": [
                f"GAN_{i:04d}"
                for i in range(n_curves)
            ],
            "source": ["GAN"] * n_curves
        })

        # Create flux columns
        flux_df = pd.DataFrame(
            flux,
            columns=[
                f"flux_{i:04d}"
                for i in range(n_bins)
            ]
        )

        # Create uncertainty columns
        flux_err_df = pd.DataFrame(
            flux_err,
            columns=[
                f"flux_err_{i:04d}"
                for i in range(n_bins)
            ]
        )

        # Combine metadata and generated signals
        results = pd.concat(
            [metadata, flux_df, flux_err_df],
            axis=1
        )

        # Save CSV in the current working directory
        if save_path is None:

            save_path = (
                Path.cwd() / "GAN_generated_curves.csv"
            )

        save_path = Path(save_path)

        results.to_csv(
            save_path,
            index=False
        )

        print("\nGENERATED LIGHT CURVES")
        print(f"Number of curves: {n_curves}")
        print(f"Number of bins: {n_bins}")
        print(f"CSV shape: {results.shape}")
        print(f"CSV saved to: {save_path}")

        return results


    def plot_signals(self, results, n_signals=8):

        # Select flux and uncertainty columns
        flux_cols = [
            f"flux_{i:04d}"
            for i in range(self.target_bins)
        ]

        flux_err_cols = [
            f"flux_err_{i:04d}"
            for i in range(self.target_bins)
        ]

        # Select the first generated curves
        n_signals = min(
            n_signals, len(results)
        )

        selected = results.iloc[:n_signals]

        flux = selected[flux_cols].to_numpy()

        flux_err = selected[
            flux_err_cols
        ].to_numpy()

        # Create subplots
        n_cols = 2

        n_rows = (
            n_signals + n_cols - 1
        ) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 3 * n_rows),
            squeeze=False
        )

        axes = axes.ravel()

        bins = np.arange(self.target_bins)

        for i in range(n_signals):

            ax = axes[i]

            ax.plot(
                bins,
                flux[i],
                linewidth=0.8,
                label="Flux"
            )

            ax.fill_between(
                bins,
                flux[i] - flux_err[i],
                flux[i] + flux_err[i],
                alpha=0.3,
                color="tab:orange",
                label="Flux uncertainty"
            )

            ax.set_title(
                selected.iloc[i]["toi_name"]
            )

            ax.set_xlabel("Bin")
            ax.set_ylabel("Flux")

            ax.legend()
            ax.grid()

        # Hide unused axes
        for ax in axes[n_signals:]:
            ax.set_visible(False)

        fig.suptitle(
            "GAN generated light curves",
            fontsize=14
        )

        fig.tight_layout()

        return fig


    def save_weights(self, save_path=None):

        if not self.trained:
            raise RuntimeError(
                "The GAN must be trained before saving weights."
            )

        if save_path is None:

            save_path = (
                Path.cwd() / "GAN_weights.pth"
            )

        # Save final weights and normalization parameters
        torch.save({

            "generator": {
                key: value.detach().cpu()
                for key, value
                in self.generator.state_dict().items()
            },

            "discriminator": {
                key: value.detach().cpu()
                for key, value
                in self.discriminator.state_dict().items()
            },

            "normalization": self.normalization,

            "GAN_params": self.GAN_params

        }, save_path)

        print(
            f"GAN weights saved to: {save_path}"
        )


    def run(self, save=False):

        output_path = Path.cwd()

        # Train the GAN
        self.train_model()

        # Plot Generator and Discriminator losses
        loss_figure = self.plot_losses()

        # Generate and save synthetic light curves
        results = self.generate_csv(
            save_path=(
                output_path / "GAN_generated_curves.csv"
            )
        )

        # Plot the first eight generated light curves
        signals_figure = self.plot_signals(
            results,
            n_signals=8
        )

        # Optional saving
        if save:

            loss_figure.savefig(
                output_path / "GAN_losses.png",
                dpi=300,
                bbox_inches="tight"
            )

            signals_figure.savefig(
                output_path / "GAN_generated_signals.png",
                dpi=300,
                bbox_inches="tight"
            )

            self.save_weights(
                output_path / "GAN_weights.pth"
            )

            print(
                f"All results saved to: {output_path}"
            )

        # Display figures
        plt.show()

        return results
