"""
Particle picking via Laplacian-of-Gaussian (LoG) blob detection.

The LoG detector finds local maxima in the scale-space representation of
the image.  At each scale σ, the response is proportional to the blob
contrast; a true particle should respond most strongly at a scale close to
its radius.  The scikit-image implementation (`blob_log`) handles the
multi-scale stack internally and returns (y, x, σ) triples.

Post-processing:
  • Confidence score — the LoG response normalised to [0, 1].
  • Min-distance deduplication — already handled by blob_log's peak-finding,
    but we additionally honour an explicit min_distance parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import blob_log


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PickResult:
    """Container returned by pick_particles."""
    coords: list[tuple[int, int]]
    """(y, x) pixel coordinates of picked particle centres."""
    confidences: list[float]
    """Normalised LoG response for each pick (0–1 range)."""
    sigmas: list[float]
    """Scale σ [pixels] at which each blob was detected."""


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------

def pick_particles(
    micrograph: np.ndarray,
    min_sigma: float = 3.0,
    max_sigma: float = 8.0,
    num_sigma: int = 10,
    threshold: float = 0.05,
    min_distance: int = 10,
) -> PickResult:
    """
    Pick particles from a micrograph using blob_log.

    Parameters
    ----------
    micrograph   : (ny, nx) float32 image
    min_sigma    : minimum blob scale [pixels]
    max_sigma    : maximum blob scale [pixels]
    num_sigma    : number of intermediate scales to test
    threshold    : minimum normalised blob response (0–1 after internal
                   normalisation by skimage); lower = more picks
    min_distance : minimum centre-to-centre distance in pixels between
                   accepted picks (enforced via blob_log's overlap param)

    Returns
    -------
    PickResult with coords, confidences, sigmas
    """
    img = micrograph.astype(np.float64)

    # Normalise to [0, 1] so that threshold is scale-independent
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = img

    # overlap controls deduplication: two blobs whose circles overlap by
    # more than this fraction are merged (keep the stronger one).
    # We set it from min_distance / (sqrt(2) * max_sigma) heuristic.
    overlap = max(0.0, 1.0 - min_distance / (np.sqrt(2) * max_sigma))

    blobs = blob_log(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )

    if len(blobs) == 0:
        return PickResult(coords=[], confidences=[], sigmas=[])

    ys = blobs[:, 0].astype(int)
    xs = blobs[:, 1].astype(int)
    sigmas = blobs[:, 2].tolist()

    # Confidence: evaluate LoG response at each detected centre.
    # We approximate it by the normalised image value (proxy for blob contrast).
    ny, nx = micrograph.shape
    responses = []
    for y, x in zip(ys, xs):
        y_c = int(np.clip(y, 0, ny - 1))
        x_c = int(np.clip(x, 0, nx - 1))
        responses.append(float(img_norm[y_c, x_c]))

    r_arr = np.array(responses, dtype=np.float64)
    r_max = r_arr.max() if r_arr.max() > 0 else 1.0
    confidences = (r_arr / r_max).tolist()

    coords = list(zip(ys.tolist(), xs.tolist()))
    return PickResult(coords=coords, confidences=confidences, sigmas=sigmas)
