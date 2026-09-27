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


class TestPickerNMS:
    @staticmethod
    def _two_blobs(gap: int) -> np.ndarray:
        """A strong and a weaker Gaussian blob `gap` px apart on a flat background."""
        yy, xx = np.mgrid[:128, :128]
        g = lambda cy, cx, a: a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
        return (g(64, 50, 1.0) + g(64, 50 + gap, 0.5)).astype(np.float32)

    def test_default_has_no_suppression(self):
        img = _make_micrograph(128)
        a = pick_particles(img, threshold=0.01)
        b = pick_particles(img, threshold=0.01, nms_radius=None)
        assert a.coords == b.coords and a.sigmas == b.sigmas

    def test_suppressed_picks_are_a_subset_and_separated(self):
        img = _make_micrograph(128)
        full = pick_particles(img, threshold=0.01)
        nms = pick_particles(img, threshold=0.01, nms_radius=15)
        assert set(nms.coords) <= set(full.coords)
        c = np.array(nms.coords, float)
        if len(c) > 1:
            d = np.linalg.norm(c[:, None] - c[None], axis=2)
            np.fill_diagonal(d, np.inf)
            assert d.min() > 15

    def test_keeps_the_stronger_of_two_nearby_blobs(self):
        img = self._two_blobs(gap=14)
        both = pick_particles(img, threshold=0.05, nms_radius=5)
        one = pick_particles(img, threshold=0.05, nms_radius=20)
        assert len(both.coords) == 2
        assert len(one.coords) == 1
        y, x = one.coords[0]
        assert abs(y - 64) <= 1 and abs(x - 50) <= 1

    @pytest.mark.parametrize("gap, expected", [(14, 1), (17, 1), (19, 1), (22, 2), (26, 2), (30, 2)])
    def test_radius_merges_only_within_its_distance(self, gap, expected):
        """Two separated blobs: nms_radius=20 keeps one within 20 px and both beyond it. See
        benchmarks/nms_separation.py for the same question on simulated particles."""
        img = self._two_blobs(gap=gap)
        assert len(pick_particles(img, threshold=0.05).coords) == 2  # both detected without suppression
        assert len(pick_particles(img, threshold=0.05, nms_radius=20).coords) == expected

    def test_lists_stay_aligned_after_suppression(self):
        pk = pick_particles(_make_micrograph(128), threshold=0.01, nms_radius=15)
        assert len(pk.coords) == len(pk.confidences) == len(pk.sigmas)

    def test_pipeline_default_on_readme_example(self):
        """Regression check on the README example (seed 42): all 32 particles are still
        found, and suppression cuts the 137 unsuppressed picks to 54."""
        import warnings
        from processor.pipeline import Pipeline, ProcessConfig
        from simulator.generator import SimulationConfig, generate_micrograph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mic, truth = generate_micrograph(SimulationConfig(n_particles=100, seed=42))
        assert ProcessConfig().nms_radius == 20.0
        picks = np.array(Pipeline(ProcessConfig()).run(mic).picks.coords, float)
        d = np.linalg.norm(picks[:, None] - np.array(truth, float)[None], axis=2)
        assert len(picks) == 54
        assert (d.min(axis=0) <= 10).all()  # every true particle has a pick within 10 px


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
