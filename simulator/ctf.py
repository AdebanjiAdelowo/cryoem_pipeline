"""
Contrast Transfer Function (CTF) simulation.

The CTF describes how a transmission electron microscope modulates spatial
frequencies in the image.  It arises from the interplay of defocus (which
shifts the focus plane of the lens) and spherical aberration (Cs).  The
CTF oscillates between −1 and +1 as a function of spatial frequency k,
meaning that some frequencies are inverted, some attenuated, and some
(at the CTF zeroes) completely lost.

Standard phase-contrast CTF (with amplitude contrast Q):
    γ(k) = π·λ·Δf·k² + (π/2)·Cs·λ³·k⁴
    CTF(k) = −√(1−Q²)·sin γ(k) + Q·cos γ(k)

where
    k   = spatial frequency [Å⁻¹]
    λ   = relativistic electron wavelength [Å]
    Δf  = defocus [Å], positive = underfocus (standard convention)
    Cs  = spherical aberration [Å]
    Q   = amplitude contrast fraction (~0.07 for protein)
"""

import numpy as np


def electron_wavelength(voltage_kv: float) -> float:
    """
    Relativistic de Broglie wavelength of electrons [Å].

    Parameters
    ----------
    voltage_kv : accelerating voltage in kilovolts

    Returns
    -------
    lambda_angstrom : wavelength in Ångströms
        At 300 kV → ~0.01969 Å
    """
    v = voltage_kv * 1e3  # kV → V
    # Relativistic formula: λ = h / sqrt(2·m₀·eV·(1 + eV/(2·m₀c²)))
    # Simplified numerically (λ in Å, V in volts):
    return 12.2643 / np.sqrt(v * (1.0 + 0.97845e-6 * v))


def compute_ctf_2d(
    shape: tuple[int, int],
    defocus_um: float,
    pixel_size_a: float,
    voltage_kv: float = 300.0,
    cs_mm: float = 2.0,
    amplitude_contrast: float = 0.07,
) -> np.ndarray:
    """
    Compute a 2D CTF array for an image of the given shape.

    Parameters
    ----------
    shape            : (ny, nx) image dimensions in pixels
    defocus_um       : defocus in microns (positive = underfocus)
    pixel_size_a     : pixel size in Ångströms
    voltage_kv       : accelerating voltage [kV]
    cs_mm            : spherical aberration coefficient [mm]
    amplitude_contrast: fraction of amplitude contrast (Q), typically 0.07

    Returns
    -------
    ctf : (ny, nx) float32 array with values in [−1, 1]
    """
    ny, nx = shape
    lambda_a  = electron_wavelength(voltage_kv)      # Å
    defocus_a = defocus_um * 1e4                      # μm → Å
    cs_a      = cs_mm * 1e7                           # mm → Å

    # Spatial frequency grids in Å⁻¹
    ky = np.fft.fftfreq(ny) / pixel_size_a
    kx = np.fft.fftfreq(nx) / pixel_size_a
    Kx, Ky = np.meshgrid(kx, ky)
    k2 = Kx**2 + Ky**2                               # Å⁻²

    # Phase of the wave aberration function
    gamma = (np.pi * lambda_a * defocus_a * k2
             + 0.5 * np.pi * cs_a * lambda_a**3 * k2**2)

    q = amplitude_contrast
    ctf = (-np.sqrt(1.0 - q**2) * np.sin(gamma)
           + q * np.cos(gamma)).astype(np.float32)
    return ctf


def apply_ctf(
    image: np.ndarray,
    defocus_um: float,
    pixel_size_a: float = 2.0,
    voltage_kv: float = 300.0,
    cs_mm: float = 2.0,
) -> np.ndarray:
    """
    Convolve a 2D image with the CTF in Fourier space.

    Multiplication in Fourier space is equivalent to convolution in real space.
    The CTF is real-valued (ignoring the imaginary part of the scattering
    potential) so no phase information is discarded here.

    Parameters
    ----------
    image        : (ny, nx) float32 input image
    defocus_um   : defocus in microns
    pixel_size_a : pixel size in Å (default 2.0 Å/px, typical for cryoEM)
    voltage_kv   : accelerating voltage [kV]
    cs_mm        : spherical aberration [mm]

    Returns
    -------
    ctf_image : (ny, nx) float32 array
    """
    ctf = compute_ctf_2d(image.shape, defocus_um, pixel_size_a, voltage_kv, cs_mm)
    ft = np.fft.fft2(image.astype(np.float32))
    return np.real(np.fft.ifft2(ft * ctf)).astype(np.float32)
