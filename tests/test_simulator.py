"""
Unit tests for the simulator package.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from simulator.ctf import apply_ctf, compute_ctf_2d, electron_wavelength
from simulator.generator import (
    SimulationConfig,
    generate_micrograph,
    generate_particle_stack,
)
from simulator.noise import add_gaussian_noise, add_noise, add_poisson_noise
from simulator.projector import make_particle_volume, project, random_rotation_matrix


# ---------------------------------------------------------------------------
# projector
# ---------------------------------------------------------------------------

class TestProjector:
    def test_volume_shape(self):
        vol = make_particle_volume(32)
        assert vol.shape == (32, 32, 32)

    def test_volume_range(self):
        vol = make_particle_volume(32)
        assert vol.min() >= 0.0
        assert abs(vol.max() - 1.0) < 1e-5

    def test_volume_dtype(self):
        assert make_particle_volume(32).dtype == np.float32

    def test_rotation_matrix_shape(self):
        R = random_rotation_matrix()
        assert R.shape == (3, 3)

    def test_rotation_matrix_orthonormal(self):
        R = random_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_projection_shape(self):
        vol = make_particle_volume(32)
        R = random_rotation_matrix()
        proj = project(vol, R)
        assert proj.shape == (32, 32)

    def test_projection_dtype(self):
        vol = make_particle_volume(32)
        proj = project(vol, random_rotation_matrix())
        assert proj.dtype == np.float32

    def test_projections_differ_by_rotation(self):
        """Two random projections should (almost certainly) not be identical."""
        vol = make_particle_volume(32)
        p1 = project(vol, random_rotation_matrix())
        p2 = project(vol, random_rotation_matrix())
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# CTF
# ---------------------------------------------------------------------------

class TestCTF:
    def test_wavelength_at_300kv(self):
        lam = electron_wavelength(300.0)
        assert abs(lam - 0.01969) < 1e-4

    def test_ctf_2d_shape(self):
        ctf = compute_ctf_2d((64, 64), defocus_um=2.0, pixel_size_a=2.0)
        assert ctf.shape == (64, 64)

    def test_ctf_2d_range(self):
        ctf = compute_ctf_2d((64, 64), defocus_um=2.0, pixel_size_a=2.0)
        assert ctf.min() >= -1.01
        assert ctf.max() <= 1.01

    def test_apply_ctf_preserves_shape(self):
        img = np.random.rand(64, 64).astype(np.float32)
        out = apply_ctf(img, defocus_um=2.0, pixel_size_a=2.0)
        assert out.shape == img.shape
        assert out.dtype == np.float32

    def test_apply_ctf_changes_image(self):
        img = np.random.rand(64, 64).astype(np.float32)
        out = apply_ctf(img, defocus_um=2.0, pixel_size_a=2.0)
        assert not np.allclose(img, out)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class TestNoise:
    def test_gaussian_noise_shape(self):
        img = np.ones((32, 32), dtype=np.float32)
        assert add_gaussian_noise(img, snr=0.1).shape == (32, 32)

    def test_gaussian_noise_increases_variance(self):
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        noisy = add_gaussian_noise(img, snr=0.1)
        assert noisy.var() > img.var()

    def test_poisson_noise_shape(self):
        img = np.random.rand(32, 32).astype(np.float32)
        assert add_poisson_noise(img).shape == (32, 32)

    def test_add_noise_no_poisson(self):
        img = np.random.rand(64, 64).astype(np.float32)
        out = add_noise(img, snr=0.1, poisson=False)
        assert out.shape == img.shape

    def test_add_noise_with_poisson(self):
        img = (np.random.rand(64, 64) + 0.1).astype(np.float32)
        out = add_noise(img, snr=0.1, poisson=True)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def setup_method(self):
        self.cfg = SimulationConfig(
            n_particles=10,
            box_size=32,
            micrograph_size=256,
            seed=42,
        )

    def test_micrograph_shape(self):
        mic, coords = generate_micrograph(self.cfg)
        assert mic.shape == (256, 256)

    def test_micrograph_dtype(self):
        mic, _ = generate_micrograph(self.cfg)
        assert mic.dtype == np.float32

    def test_coords_within_bounds(self):
        _, coords = generate_micrograph(self.cfg)
        size = self.cfg.micrograph_size
        for y, x in coords:
            assert 0 <= y < size
            assert 0 <= x < size

    def test_stack_shape(self):
        stack = generate_particle_stack(self.cfg)
        assert stack.ndim == 3
        assert stack.shape[1] == self.cfg.box_size
        assert stack.shape[2] == self.cfg.box_size

    def test_stack_n_particles(self):
        stack = generate_particle_stack(self.cfg)
        assert stack.shape[0] == self.cfg.n_particles

    def test_reproducibility(self):
        mic1, c1 = generate_micrograph(self.cfg)
        mic2, c2 = generate_micrograph(self.cfg)
        assert np.allclose(mic1, mic2)
        assert c1 == c2
