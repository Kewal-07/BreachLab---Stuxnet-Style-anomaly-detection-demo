"""Autoencoder detector (PyTorch).

A small fully-connected autoencoder trained to reconstruct *windows* of normal
telemetry. The reconstruction error is the anomaly score: the network learns
the manifold of normal behaviour, so anything it has never seen -- an attack --
reconstructs poorly and scores high.

Design choices
--------------
* **Windows, not single timesteps.** Each input is ``window_size`` consecutive
  timesteps flattened, so the network can learn *temporal* structure (e.g. the
  thermal lag), not just instantaneous correlations.
* **Symmetric encoder/decoder** ``input -> hidden -> latent -> hidden -> input``
  with a latent bottleneck that forces the network to compress normal patterns.
* Fully **seeded** (Python/NumPy/Torch) for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import DetectorConfig
from ..logging_utils import get_logger
from .base import Detector, Standardizer
from .windowing import align_window_scores, make_windows

logger = get_logger("detectors.autoencoder")


class _AutoEncoderNet(nn.Module):
    """input -> hidden -> latent -> hidden -> input, with ReLU activations."""

    def __init__(self, input_dim: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(Detector):
    """Reconstruction-error anomaly detector over sliding windows."""

    name = "Autoencoder"

    def __init__(self, config: DetectorConfig, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.scaler_ = Standardizer()
        self.net_: _AutoEncoderNet | None = None
        self.window_ = config.window_size

    def _build_windows(self, X: np.ndarray, fit_scaler: bool) -> np.ndarray:
        Xs = self.scaler_.fit_transform(X) if fit_scaler else self.scaler_.transform(X)
        return make_windows(Xs, self.window_)

    def _fit(self, X: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        windows = self._build_windows(X, fit_scaler=True)
        input_dim = windows.shape[1]
        tensor = torch.from_numpy(windows).float()
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.ae_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.net_ = _AutoEncoderNet(input_dim, self.cfg.ae_hidden, self.cfg.ae_latent)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.cfg.ae_lr)
        loss_fn = nn.MSELoss()

        self.net_.train()
        for epoch in range(self.cfg.ae_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.net_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "AE epoch %d/%d  loss=%.4f",
                    epoch + 1,
                    self.cfg.ae_epochs,
                    epoch_loss / len(dataset),
                )

    def _score(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        windows = self._build_windows(X, fit_scaler=False)
        tensor = torch.from_numpy(windows).float()
        with torch.no_grad():
            recon = self.net_(tensor)
            # Per-window mean squared reconstruction error.
            errors = ((recon - tensor) ** 2).mean(dim=1).numpy()
        return align_window_scores(errors, X.shape[0], self.window_)
