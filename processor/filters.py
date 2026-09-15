"""
Fourier-space filtering utilities for cryoEM images.

Two filters are provided:
  • bandpass_filter  — keeps only spatial frequencies between low_freq and
    high_freq (in Å⁻¹), attenuating everything outside that band.  This
    suppresses both low-frequency background gradients (carbon film,
    charging) and high-frequency detector noise.
  • wiener_ctf_correction — inverts the CTF using the Wiener estimator
        W(k) = CTF*(k) / (|CTF(k)|² + 1/SNR)
    which regularises the inversion near CTF zeroes where the signal has
    been destroyed.  The corrected image has its Fourier amplitudes
    restored to the pre-CTF spectrum.
"""

import numpy as np

from simulator.ctf import compute_ctf_2d


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    image: np.ndarray,
    low_freq: float,
    high_freq: float,
    pixel_size_a: float = 2.0,
) -> np.ndarray:
    """
    Apply a hard-edged bandpass filter in Fourier space.

    Frequencies outside [low_freq, high_freq] are zeroed.  The DC
    component (k = 0) is preserved only when low_freq == 0.

    Parameters
    ----------
    image        : (ny, nx) float32 input image
    low_freq     : lower frequency cutoff  [Å⁻¹]  (0 = keep DC)
    high_freq    : upper frequency cutoff  [Å⁻¹]  (Nyquist = 1/(2*pixel_size_a))
    pixel_size_a : pixel size in Ångströms

    Returns
    -------
    filtered : (ny, nx) float32 array
    """
    ny, nx = image.shape
    ky = np.fft.fftfreq(ny) / pixel_size_a
    kx = np.fft.fftfreq(nx) / pixel_size_a
    Kx, Ky = np.meshgrid(kx, ky)
    k = np.sqrt(Kx**2 + Ky**2)

    mask = (k >= low_freq) & (k <= high_freq)

    ft = np.fft.fft2(image.astype(np.float32))
    ft_filtered = ft * mask
    return np.real(np.fft.ifft2(ft_filtered)).astype(np.float32)


# ---------------------------------------------------------------------------
# Wiener CTF correction
# ---------------------------------------------------------------------------

def wiener_ctf_correction(
    image: np.ndarray,
    defocus_um: float,
    pixel_size_a: float = 2.0,
    voltage_kv: float = 300.0,
    cs_mm: float = 2.0,
    snr_estimate: float = 0.1,
) -> np.ndarray:
    """
    Correct for CTF phase flips using the Wiener filter.

    The Wiener estimator in frequency space is:
        W(k) = CTF*(k) / (|CTF(k)|² + 1/SNR)
    Multiplying the Fourier transform of the image by W(k) then
    back-transforming gives an estimate of the CTF-free image.

    Because the CTF is real-valued here, CTF* = CTF.

    Parameters
    ----------
    image        : (ny, nx) float32 input image
    defocus_um   : defocus used when acquiring / simulating the image [μm]
    pixel_size_a : pixel size [Å]
    voltage_kv   : accelerating voltage [kV]
    cs_mm        : spherical aberration [mm]
    snr_estimate : estimate of the signal-to-noise ratio; controls
                   regularisation strength (smaller → heavier smoothing)

    Returns
    -------
    corrected : (ny, nx) float32 array
    """
    ctf = compute_ctf_2d(
        image.shape, defocus_um, pixel_size_a, voltage_kv, cs_mm
    ).astype(np.float64)

    # Wiener filter: CTF / (CTF² + ε)   where ε = 1/SNR
    epsilon = 1.0 / max(snr_estimate, 1e-6)
    wiener = ctf / (ctf**2 + epsilon)

    ft = np.fft.fft2(image.astype(np.float64))
    corrected = np.real(np.fft.ifft2(ft * wiener))
    return corrected.astype(np.float32)
