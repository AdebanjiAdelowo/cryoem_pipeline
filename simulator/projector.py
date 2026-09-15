"""
3D particle model and projection engine.

A cryoEM image is essentially a 2D projection of a 3D molecular density,
blurred by the CTF and buried in noise.  Here we represent the particle as
an asymmetric Gaussian blob so that different orientations yield visibly
different projections.
"""

import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation


def make_particle_volume(box_size: int) -> np.ndarray:
    """
    Build a 3D asymmetric particle density on a (box_size)³ grid.

    The density is a sum of two Gaussians with different widths and
    a small offset between them, making it orientation-sensitive.

    Parameters
    ----------
    box_size : side length in voxels

    Returns
    -------
    volume : (box_size, box_size, box_size) float32 array, values in [0, 1]
    """
    c = box_size / 2.0
    r = box_size / 4.0  # characteristic radius

    z, y, x = np.mgrid[0:box_size, 0:box_size, 0:box_size].astype(np.float32)
    z -= c; y -= c; x -= c

    # Primary elongated Gaussian (σx > σy > σz to break symmetry)
    primary = np.exp(
        -(x**2 / (0.80 * r) ** 2
          + y**2 / (0.50 * r) ** 2
          + z**2 / (0.30 * r) ** 2)
    )

    # Satellite blob offset from centre — acts like a second protein domain
    satellite = 0.5 * np.exp(
        -((x - 0.40 * r) ** 2
          + (y - 0.30 * r) ** 2
          + z**2)
        / (0.20 * r) ** 2
    )

    volume = (primary + satellite).astype(np.float32)
    volume /= volume.max()
    return volume


def random_rotation_matrix(
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample a uniformly random rotation from SO(3).

    Parameters
    ----------
    rng : optional numpy Generator; if provided, rotation is drawn
          reproducibly from it.  If None, scipy uses its own global state.

    Returns
    -------
    R : (3, 3) float64 rotation matrix
    """
    random_state = int(rng.integers(0, 2**31)) if rng is not None else None
    return Rotation.random(random_state=random_state).as_matrix()


def project(volume: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Project a 3D density to 2D by rotating the volume then summing along z.

    This implements the central-projection approximation used in cryoEM:
    the electron beam travels along z and the detector records the
    line integral of the density.

    Parameters
    ----------
    volume   : (D, H, W) float32 density array
    rotation : (3, 3) rotation matrix — rotates the particle before projection

    Returns
    -------
    projection : (H, W) float32 array
    """
    center = np.array(volume.shape, dtype=float) / 2.0

    # scipy.ndimage.affine_transform maps output coords to input coords via:
    #   input_coord = matrix @ output_coord + offset
    # To rotate the *data* by R, we must pull back using R^T = R^{-1}.
    offset = center - rotation.T @ center

    rotated = affine_transform(
        volume,
        rotation.T,
        offset=offset,
        order=1,         # bilinear — fast and adequate
        mode="constant",
        cval=0.0,
    )

    # Integrate along the beam direction (axis 0 = z)
    return rotated.sum(axis=0).astype(np.float32)
