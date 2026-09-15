"""
Noise models for cryoEM image simulation.

Real cryoEM data is dominated by noise — typical SNR is 0.05–0.2.
Two noise sources matter most:
  • Gaussian noise — models detector readout noise and inelastic scattering
  • Poisson (shot) noise — models the statistical nature of electron counting;
    variance equals the mean electron count per pixel
"""

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    snr: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to achieve a target signal-to-noise ratio.

    SNR is defined as  σ²_signal / σ²_noise.
    For cryoEM, SNR ≈ 0.1 means noise standard deviation is ~3× the
    signal standard deviation.

    Parameters
    ----------
    image : (ny, nx) float32 input image
    snr   : desired signal-to-noise ratio (e.g. 0.1 for cryoEM)
    rng   : optional numpy Generator for reproducibility

    Returns
    -------
    noisy : (ny, nx) float32 array
    """
    signal_var = float(np.var(image))
    if signal_var < 1e-12:
        signal_var = 1.0  # avoid division by zero on flat images
    noise_std = np.sqrt(signal_var / snr)
    _rng = rng if rng is not None else np.random.default_rng()
    noise = _rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    return (image + noise).astype(np.float32)


def add_poisson_noise(
    image: np.ndarray,
    electrons_per_pixel: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Poisson (shot) noise to simulate electron-counting statistics.

    The image is rescaled so that its mean corresponds to
    `electrons_per_pixel` expected counts, Poisson-sampled, then
    rescaled back to the original intensity range.

    Parameters
    ----------
    image               : (ny, nx) float32 input image
    electrons_per_pixel : expected electron dose per pixel
    rng                 : optional numpy Generator for reproducibility

    Returns
    -------
    noisy : (ny, nx) float32 array
    """
    img_min = float(image.min())
    shifted = image - img_min                          # shift to non-negative
    mean_signal = float(shifted.mean())
    if mean_signal < 1e-12:
        return image.copy()

    scale = electrons_per_pixel / mean_signal
    _rng = rng if rng is not None else np.random.default_rng()
    counts = _rng.poisson(
        (shifted * scale).astype(np.float64)
    ).astype(np.float32)
    return (counts / scale + img_min).astype(np.float32)


def add_noise(
    image: np.ndarray,
    snr: float = 0.1,
    poisson: bool = False,
    electrons_per_pixel: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply the complete noise model: Gaussian, and optionally Poisson.

    Parameters
    ----------
    image               : (ny, nx) float32 input image
    snr                 : target Gaussian SNR
    poisson             : if True, also add Poisson shot noise
    electrons_per_pixel : dose parameter for Poisson model
    rng                 : optional numpy Generator for reproducibility

    Returns
    -------
    noisy : (ny, nx) float32 array
    """
    noisy = add_gaussian_noise(image, snr, rng=rng)
    if poisson:
        noisy = add_poisson_noise(noisy, electrons_per_pixel, rng=rng)
    return noisy
