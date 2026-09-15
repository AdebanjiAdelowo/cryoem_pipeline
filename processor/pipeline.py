"""
Processing pipeline that chains: filter → pick → align.

The Pipeline class runs each stage in sequence and exposes a status
attribute (pending / running / done / failed) so the API layer can
report progress without threading.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .aligner import ClassAverageResult, class_average
from .filters import bandpass_filter, wiener_ctf_correction
from .picker import PickResult, pick_particles


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProcessConfig:
    """Tunable parameters for the processing pipeline."""

    # --- bandpass ---
    low_freq: float = 0.02
    """Low-frequency cutoff [Å⁻¹]  (removes background gradients)."""
    high_freq: float = 0.25
    """High-frequency cutoff [Å⁻¹] (removes high-freq noise)."""

    # --- CTF correction ---
    apply_wiener: bool = True
    """Whether to apply Wiener CTF correction before picking."""
    defocus_um: float = 2.0
    """Defocus estimate for Wiener correction [μm]."""
    pixel_size_a: float = 2.0
    voltage_kv: float = 300.0
    cs_mm: float = 2.0
    snr_estimate: float = 0.1

    # --- picking ---
    min_sigma: float = 3.0
    max_sigma: float = 8.0
    num_sigma: int = 10
    pick_threshold: float = 0.05
    min_distance: int = 10

    # --- class averaging ---
    box_size: int = 64
    n_classes: int = 5
    random_state: Optional[int] = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    filtered_micrograph: np.ndarray
    """Bandpass- (and optionally Wiener-) filtered micrograph."""
    picks: PickResult
    """Particle picks from blob_log."""
    class_avg: ClassAverageResult
    """k-means class averages."""
    n_picked: int
    """Number of particles picked."""
    n_classes: int
    """Number of classes requested (actual may be lower if n_picked < n_classes)."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Stateful processing pipeline.

    Usage
    -----
    pipeline = Pipeline(config)
    result   = pipeline.run(micrograph)
    print(pipeline.status)   # PipelineStatus.DONE
    """

    def __init__(self, config: ProcessConfig | None = None) -> None:
        self.config = config or ProcessConfig()
        self.status: PipelineStatus = PipelineStatus.PENDING
        self.result: Optional[ProcessResult] = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------
    def run(self, micrograph: np.ndarray) -> ProcessResult:
        """
        Execute the full pipeline on `micrograph`.

        Sets self.status to RUNNING, then DONE (or FAILED on exception).

        Parameters
        ----------
        micrograph : (ny, nx) float32 micrograph

        Returns
        -------
        ProcessResult
        """
        self.status = PipelineStatus.RUNNING
        self.error = None
        cfg = self.config

        try:
            # 1. Bandpass filter
            filtered = bandpass_filter(
                micrograph, cfg.low_freq, cfg.high_freq, cfg.pixel_size_a
            )

            # 2. Optional Wiener CTF correction
            if cfg.apply_wiener:
                filtered = wiener_ctf_correction(
                    filtered,
                    cfg.defocus_um,
                    cfg.pixel_size_a,
                    cfg.voltage_kv,
                    cfg.cs_mm,
                    cfg.snr_estimate,
                )

            # 3. Particle picking
            picks = pick_particles(
                filtered,
                min_sigma=cfg.min_sigma,
                max_sigma=cfg.max_sigma,
                num_sigma=cfg.num_sigma,
                threshold=cfg.pick_threshold,
                min_distance=cfg.min_distance,
            )

            # 4. Class averaging
            avg = class_average(
                filtered,
                picks.coords,
                box_size=cfg.box_size,
                n_classes=cfg.n_classes,
                random_state=cfg.random_state,
            )

            self.result = ProcessResult(
                filtered_micrograph=filtered,
                picks=picks,
                class_avg=avg,
                n_picked=len(picks.coords),
                n_classes=len(avg.class_averages),
            )
            self.status = PipelineStatus.DONE
            return self.result

        except Exception:
            self.error = traceback.format_exc()
            self.status = PipelineStatus.FAILED
            raise
