"""
Main simulation entry point.

Generates:
  1. A synthetic micrograph — a large image containing N particles placed at
     random non-overlapping positions, CTF-modulated, and buried in noise.
  2. A particle stack — N individually simulated particles each with a
     different random orientation, defocus, and noise realisation.

Both are saved as MRC files (the standard format in cryoEM) and as PNG
images for quick visual inspection.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from .ctf import apply_ctf
from .noise import add_noise
from .projector import make_particle_volume, project, random_rotation_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All parameters that drive the simulation."""

    n_particles: int = 100
    """Number of particles to generate / place in the micrograph."""

    box_size: int = 64
    """Particle box size in pixels (power of 2 recommended)."""

    defocus_um: float = 2.0
    """Central defocus value in microns (positive = underfocus)."""

    defocus_spread_um: float = 0.5
    """±Spread of defocus across particles [μm]."""

    snr: float = 0.1
    """Gaussian signal-to-noise ratio (0.1 is typical for cryoEM)."""

    pixel_size_a: float = 2.0
    """Pixel size in Ångströms."""

    voltage_kv: float = 300.0
    """Accelerating voltage [kV]."""

    cs_mm: float = 2.0
    """Spherical aberration coefficient [mm]."""

    add_poisson: bool = False
    """Whether to add Poisson (shot) noise on top of Gaussian noise."""

    micrograph_size: int = 512
    """Side length of the synthetic micrograph in pixels."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _place_particles(
    micrograph_size: int,
    box_size: int,
    n_particles: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Sample non-overlapping (y, x) centre coordinates for particles.

    A simple rejection-sampling loop: each candidate is rejected if it
    overlaps (within one box_size) with any already-placed particle.
    """
    margin = box_size // 2 + 4
    coords: list[tuple[int, int]] = []
    max_attempts = n_particles * 200

    for _ in range(max_attempts):
        if len(coords) >= n_particles:
            break
        cy = int(rng.integers(margin, micrograph_size - margin))
        cx = int(rng.integers(margin, micrograph_size - margin))
        if any(
            abs(cy - ey) < box_size and abs(cx - ex) < box_size
            for ey, ex in coords
        ):
            continue
        coords.append((cy, cx))

    return coords


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------

def generate_micrograph(
    config: SimulationConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Build a synthetic micrograph by:
      1. Generating random 2D projections of the 3D particle.
      2. Placing them at random non-overlapping positions.
      3. Applying CTF to the whole field (single defocus, as in a real micrograph).
      4. Adding noise.

    Parameters
    ----------
    config : SimulationConfig

    Returns
    -------
    micrograph : (H, W) float32 array
    coords     : list of (y, x) ground-truth particle centres
    """
    rng = np.random.default_rng(config.seed)
    volume = make_particle_volume(config.box_size)
    micrograph = np.zeros(
        (config.micrograph_size, config.micrograph_size), dtype=np.float32
    )

    coords = _place_particles(
        config.micrograph_size, config.box_size, config.n_particles, rng
    )

    placed: list[tuple[int, int]] = []
    hs = config.box_size // 2
    for cy, cx in coords:
        y1, y2 = cy - hs, cy + hs
        x1, x2 = cx - hs, cx + hs
        if y1 < 0 or y2 > config.micrograph_size or x1 < 0 or x2 > config.micrograph_size:
            continue

        rotation = random_rotation_matrix(rng)
        proj = project(volume, rotation)
        # Normalise each projection to [0, 1] before embedding
        p_min, p_max = proj.min(), proj.max()
        if p_max > p_min:
            proj = (proj - p_min) / (p_max - p_min)

        micrograph[y1:y2, x1:x2] += proj
        placed.append((cy, cx))

    # CTF applied to the full micrograph (microscope property, not per-particle)
    micrograph = apply_ctf(
        micrograph, config.defocus_um, config.pixel_size_a,
        config.voltage_kv, config.cs_mm,
    )
    micrograph = add_noise(micrograph, config.snr, config.add_poisson, rng=rng)
    return micrograph, placed


def generate_particle_stack(
    config: SimulationConfig,
) -> np.ndarray:
    """
    Generate a stack of N particle images, each with:
      • A different random 3D orientation
      • A randomly drawn defocus within [defocus_um ± defocus_spread_um]
      • An independent noise realisation

    Parameters
    ----------
    config : SimulationConfig

    Returns
    -------
    stack : (N, box_size, box_size) float32 array
    """
    rng = np.random.default_rng(config.seed)
    volume = make_particle_volume(config.box_size)
    particles: list[np.ndarray] = []

    for _ in range(config.n_particles):
        rotation = random_rotation_matrix(rng)
        proj = project(volume, rotation)

        defocus = float(rng.uniform(
            max(0.1, config.defocus_um - config.defocus_spread_um),
            config.defocus_um + config.defocus_spread_um,
        ))
        proj_ctf = apply_ctf(
            proj, defocus, config.pixel_size_a, config.voltage_kv, config.cs_mm
        )
        noisy = add_noise(proj_ctf, config.snr, config.add_poisson, rng=rng)
        particles.append(noisy)

    return np.stack(particles, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_mrc(data: np.ndarray, path: str) -> None:
    """Write a 2D or 3D float32 array to an MRC file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def save_micrograph_png(
    micrograph: np.ndarray,
    coords: list[tuple[int, int]],
    path: str,
) -> None:
    """Save the micrograph as a PNG, overlaying ground-truth particle circles."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Contrast stretch for visibility
    vmin, vmax = np.percentile(micrograph, [1, 99])
    ax.imshow(micrograph, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    if coords:
        ys, xs = zip(*coords)
        ax.scatter(
            xs, ys, s=50,
            facecolors="none", edgecolors="cyan",
            linewidths=0.8, alpha=0.85,
        )
    ax.set_title(
        f"Simulated micrograph  |  {len(coords)} particles  "
        f"|  SNR={micrograph.std():.2f}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_stack_png(stack: np.ndarray, path: str, max_display: int = 25) -> None:
    """Save a montage of the first `max_display` particles."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = min(len(stack), max_display)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i < n:
            vmin, vmax = np.percentile(stack[i], [1, 99])
            ax.imshow(stack[i], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.suptitle(
        f"Particle stack — first {n} / {len(stack)}", fontsize=8, y=1.01
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(config: SimulationConfig, output_dir: str) -> dict:
    """
    Run the full simulation and write all output files.

    Parameters
    ----------
    config     : SimulationConfig
    output_dir : directory where MRC and PNG files will be written

    Returns
    -------
    result : dict with keys
        micrograph_mrc, micrograph_png, particles_mrc, particles_png,
        n_particles, ground_truth_coords
    """
    os.makedirs(output_dir, exist_ok=True)

    micrograph, coords = generate_micrograph(config)
    stack = generate_particle_stack(config)

    paths = {
        "micrograph_mrc": os.path.join(output_dir, "micrograph.mrc"),
        "micrograph_png": os.path.join(output_dir, "micrograph.png"),
        "particles_mrc":  os.path.join(output_dir, "particles.mrc"),
        "particles_png":  os.path.join(output_dir, "particles_montage.png"),
    }

    save_mrc(micrograph, paths["micrograph_mrc"])
    save_micrograph_png(micrograph, coords, paths["micrograph_png"])
    save_mrc(stack, paths["particles_mrc"])
    save_stack_png(stack, paths["particles_png"])

    return {
        **paths,
        "n_particles":        len(stack),
        "ground_truth_coords": [(int(y), int(x)) for y, x in coords],
    }
