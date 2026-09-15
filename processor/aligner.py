"""
2D class averaging via k-means clustering.

Class averaging is a central step in single-particle cryoEM analysis:
  1. Extract small boxes (patches) around each picked particle.
  2. Flatten each box to a 1D feature vector.
  3. Cluster with k-means — each cluster groups particles in similar
     orientations (or with similar CTF defoci, ice thickness, etc.).
  4. Average the patches in each cluster → class averages, which have
     much higher SNR than individual particles.

Limitations of this implementation (intentionally simple):
  • No rotational/translational alignment before clustering — would need
    cross-correlation alignment for production use.
  • k-means on raw pixels is sensitive to contrast inversions from CTF;
    in practice one would first apply Wiener correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ClassAverageResult:
    """Output of class_average."""
    class_averages: np.ndarray
    """(n_classes, box_size, box_size) float32 — mean image per class."""
    labels: list[int]
    """Cluster label (0 … n_classes-1) for each input particle."""
    class_sizes: list[int]
    """Number of particles assigned to each class."""
    class_variances: list[float]
    """Mean per-pixel variance within each class (quality metric)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_patches(
    micrograph: np.ndarray,
    coords: list[tuple[int, int]],
    box_size: int,
) -> np.ndarray:
    """
    Extract square patches from `micrograph` centred on `coords`.

    Particles too close to the edge (so that the box would exceed the
    image boundary) are silently skipped; the returned array may contain
    fewer rows than `len(coords)`.

    Returns
    -------
    patches : (n_valid, box_size, box_size) float32
    valid_indices : list[int] — which input coords were kept
    """
    hs = box_size // 2
    ny, nx = micrograph.shape
    patches: list[np.ndarray] = []
    valid_indices: list[int] = []

    for idx, (cy, cx) in enumerate(coords):
        y1, y2 = cy - hs, cy + hs
        x1, x2 = cx - hs, cx + hs
        if y1 < 0 or y2 > ny or x1 < 0 or x2 > nx:
            continue
        patch = micrograph[y1:y2, x1:x2].astype(np.float32)
        # Normalise each patch independently to zero mean, unit variance
        m, s = patch.mean(), patch.std()
        if s > 1e-8:
            patch = (patch - m) / s
        patches.append(patch)
        valid_indices.append(idx)

    if not patches:
        return np.empty((0, box_size, box_size), dtype=np.float32), []

    return np.stack(patches, axis=0), valid_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def class_average(
    micrograph: np.ndarray,
    coords: list[tuple[int, int]],
    box_size: int = 64,
    n_classes: int = 5,
    random_state: int | None = 42,
) -> ClassAverageResult:
    """
    Cluster picked particles and compute per-class averages.

    Parameters
    ----------
    micrograph   : (ny, nx) float32 micrograph
    coords       : (y, x) centres from the particle picker
    box_size     : side length of the extraction box in pixels
    n_classes    : number of k-means clusters
    random_state : seed for k-means reproducibility

    Returns
    -------
    ClassAverageResult
    """
    patches, valid_idx = _extract_patches(micrograph, coords, box_size)

    n_particles = len(patches)
    if n_particles == 0:
        empty = np.zeros((n_classes, box_size, box_size), dtype=np.float32)
        return ClassAverageResult(
            class_averages=empty,
            labels=[],
            class_sizes=[0] * n_classes,
            class_variances=[0.0] * n_classes,
        )

    k = min(n_classes, n_particles)
    features = patches.reshape(n_particles, -1)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels_arr = km.fit_predict(features)

    averages = np.zeros((k, box_size, box_size), dtype=np.float32)
    sizes: list[int] = []
    variances: list[float] = []

    for cls in range(k):
        members = patches[labels_arr == cls]
        sizes.append(int(len(members)))
        if len(members) > 0:
            averages[cls] = members.mean(axis=0)
            variances.append(float(members.var(axis=(1, 2)).mean()))
        else:
            variances.append(0.0)

    # Map valid_idx back to a full label list (unlabelled → -1)
    full_labels = [-1] * len(coords)
    for out_pos, orig_idx in enumerate(valid_idx):
        full_labels[orig_idx] = int(labels_arr[out_pos])

    return ClassAverageResult(
        class_averages=averages,
        labels=full_labels,
        class_sizes=sizes,
        class_variances=variances,
    )


def stack_from_picks(
    micrograph: np.ndarray,
    coords: list[tuple[int, int]],
    box_size: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract a particle stack from a micrograph without averaging.

    Useful for passing picked particles to downstream analysis.

    Returns
    -------
    stack        : (n_valid, box_size, box_size) float32
    valid_indices: indices into coords that were successfully extracted
    """
    patches, valid_idx = _extract_patches(micrograph, coords, box_size)
    return patches, valid_idx
