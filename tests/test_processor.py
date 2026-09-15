"""
Unit tests for the processor package.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from processor.aligner import class_average, stack_from_picks
from processor.filters import bandpass_filter, wiener_ctf_correction
from processor.picker import pick_particles
from processor.pipeline import Pipeline, PipelineStatus, ProcessConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_micrograph(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((size, size)).astype(np.float32)
    # Plant a few faint blobs so the picker can find something
    hs = 6
    for cy, cx in [(32, 32), (64, 96), (96, 48)]:
        y1, y2 = cy - hs, cy + hs
        x1, x2 = cx - hs, cx + hs
        img[y1:y2, x1:x2] += 3.0
    return img


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_bandpass_shape(self):
        img = _make_micrograph()
        out = bandpass_filter(img, low_freq=0.02, high_freq=0.25)
        assert out.shape == img.shape

    def test_bandpass_dtype(self):
        img = _make_micrograph()
        out = bandpass_filter(img, low_freq=0.02, high_freq=0.25)
        assert out.dtype == np.float32

    def test_bandpass_removes_dc_when_low_nonzero(self):
        """With low_freq > 0, DC (mean) should be suppressed."""
        img = _make_micrograph() + 10.0   # large DC offset
        out = bandpass_filter(img, low_freq=0.02, high_freq=0.25)
        assert abs(out.mean()) < abs(img.mean())

    def test_wiener_shape(self):
        img = _make_micrograph()
        out = wiener_ctf_correction(img, defocus_um=2.0)
        assert out.shape == img.shape

    def test_wiener_dtype(self):
        img = _make_micrograph()
        out = wiener_ctf_correction(img, defocus_um=2.0)
        assert out.dtype == np.float32

    def test_wiener_changes_image(self):
        img = _make_micrograph()
        out = wiener_ctf_correction(img, defocus_um=2.0)
        assert not np.allclose(img, out)


# ---------------------------------------------------------------------------
# picker
# ---------------------------------------------------------------------------

class TestPicker:
    def test_returns_pick_result(self):
        from processor.picker import PickResult
        img = _make_micrograph()
        result = pick_particles(img)
        assert isinstance(result, PickResult)

    def test_output_lengths_consistent(self):
        img = _make_micrograph()
        result = pick_particles(img)
        n = len(result.coords)
        assert len(result.confidences) == n
        assert len(result.sigmas) == n

    def test_coords_within_image(self):
        img = _make_micrograph(128)
        result = pick_particles(img)
        for y, x in result.coords:
            assert 0 <= y < 128
            assert 0 <= x < 128

    def test_finds_blobs(self):
        """The planted blobs should be detected."""
        img = _make_micrograph(128)
        result = pick_particles(img, threshold=0.01)
        assert len(result.coords) >= 1

    def test_confidences_in_range(self):
        img = _make_micrograph(128)
        result = pick_particles(img, threshold=0.01)
        for c in result.confidences:
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# aligner
# ---------------------------------------------------------------------------

class TestAligner:
    def setup_method(self):
        self.mic = _make_micrograph(256)
        self.coords = [(32, 32), (64, 96), (96, 48), (128, 128), (160, 200)]

    def test_class_average_shape(self):
        result = class_average(self.mic, self.coords, box_size=32, n_classes=3)
        assert result.class_averages.ndim == 3
        assert result.class_averages.shape[1] == 32
        assert result.class_averages.shape[2] == 32

    def test_n_classes_at_most_n_particles(self):
        result = class_average(self.mic, self.coords, box_size=32, n_classes=10)
        assert len(result.class_averages) <= len(self.coords)

    def test_labels_length(self):
        result = class_average(self.mic, self.coords, box_size=32, n_classes=3)
        assert len(result.labels) == len(self.coords)

    def test_class_sizes_sum_to_n_valid(self):
        result = class_average(self.mic, self.coords, box_size=32, n_classes=3)
        # Some coords near edge may be skipped; sum of sizes ≤ len(coords)
        assert sum(result.class_sizes) <= len(self.coords)

    def test_empty_coords(self):
        result = class_average(self.mic, [], box_size=32, n_classes=3)
        assert result.class_averages.shape[0] == 3
        assert result.labels == []

    def test_stack_from_picks(self):
        patches, valid = stack_from_picks(self.mic, self.coords, box_size=32)
        assert patches.ndim == 3
        assert patches.shape[1] == 32
        assert len(valid) == patches.shape[0]


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def setup_method(self):
        self.mic = _make_micrograph(128)
        self.cfg = ProcessConfig(
            box_size=32,
            n_classes=3,
            pick_threshold=0.01,
        )

    def test_initial_status(self):
        p = Pipeline(self.cfg)
        assert p.status == PipelineStatus.PENDING

    def test_run_sets_done(self):
        p = Pipeline(self.cfg)
        p.run(self.mic)
        assert p.status == PipelineStatus.DONE

    def test_run_returns_result(self):
        p = Pipeline(self.cfg)
        result = p.run(self.mic)
        assert result is not None

    def test_result_filtered_micrograph(self):
        p = Pipeline(self.cfg)
        result = p.run(self.mic)
        assert result.filtered_micrograph.shape == self.mic.shape

    def test_result_n_picked_consistent(self):
        p = Pipeline(self.cfg)
        result = p.run(self.mic)
        assert result.n_picked == len(result.picks.coords)

    def test_failed_status_on_bad_input(self):
        p = Pipeline(self.cfg)
        with pytest.raises(Exception):
            p.run(np.array([]))          # bad input — triggers ValueError
        assert p.status == PipelineStatus.FAILED
