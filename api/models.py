"""Pydantic request / response models for the cryoEM pipeline API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulationParams(BaseModel):
    n_particles: int = Field(100, ge=1, le=2000)
    box_size: int = Field(64, ge=16, le=256)
    defocus_um: float = Field(2.0, gt=0.0, le=10.0)
    defocus_spread_um: float = Field(0.5, ge=0.0)
    snr: float = Field(0.1, gt=0.0, le=10.0)
    pixel_size_a: float = Field(2.0, gt=0.0)
    voltage_kv: float = Field(300.0, gt=0.0)
    cs_mm: float = Field(2.0, ge=0.0)
    add_poisson: bool = False
    micrograph_size: int = Field(512, ge=64, le=4096)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

class ProcessParams(BaseModel):
    job_id: str = Field(..., description="Job ID returned by POST /simulate")
    low_freq: float = Field(0.02, ge=0.0)
    high_freq: float = Field(0.25, gt=0.0)
    apply_wiener: bool = True
    snr_estimate: float = Field(0.1, gt=0.0)
    min_sigma: float = Field(3.0, gt=0.0)
    max_sigma: float = Field(8.0, gt=0.0)
    num_sigma: int = Field(10, ge=2, le=50)
    pick_threshold: float = Field(0.05, gt=0.0, le=1.0)
    min_distance: int = Field(10, ge=1)
    n_classes: int = Field(5, ge=1, le=50)
    random_state: Optional[int] = 42


# ---------------------------------------------------------------------------
# Job / status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = ""


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    sim_params: Optional[SimulationParams] = None
    n_particles_requested: Optional[int] = None
    n_particles_placed: Optional[int] = None
    n_picked: Optional[int] = None
    n_classes: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Process result summary
# ---------------------------------------------------------------------------

class PickInfo(BaseModel):
    y: int
    x: int
    confidence: float
    sigma: float


class ClassInfo(BaseModel):
    class_id: int
    size: int
    variance: float


class ProcessResponse(BaseModel):
    job_id: str
    n_picked: int
    picks: list[PickInfo]
    classes: list[ClassInfo]
