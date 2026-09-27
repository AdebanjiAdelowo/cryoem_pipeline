"""
FastAPI route handlers for the cryoEM pipeline.

Endpoints
---------
POST /simulate          — submit a simulation job (returns job_id immediately)
POST /process           — submit a processing job for an existing sim job
GET  /status/{job_id}   — poll job status and summary statistics
GET  /results/{job_id}/micrograph  — PNG of the micrograph with pick overlays
GET  /results/{job_id}/stack       — PNG montage of the particle stack
GET  /results/{job_id}/classes     — PNG montage of class averages

All heavy work runs in a background asyncio task so the POST endpoints
are non-blocking.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from processor.pipeline import Pipeline, ProcessConfig
from simulator.generator import (
    SimulationConfig,
    generate_micrograph,
    generate_particle_stack,
    save_mrc,
)

from .models import (
    ClassInfo,
    JobResponse,
    JobStatus,
    PickInfo,
    ProcessParams,
    ProcessResponse,
    SimulationParams,
    StatusResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
# Structure per job_id:
#   status           : JobStatus
#   sim_params       : SimulationParams | None
#   micrograph       : np.ndarray | None
#   coords           : list[(y,x)] | None    ground-truth centres
#   stack            : np.ndarray | None     (N, H, W) particle stack
#   pipeline         : Pipeline | None
#   error            : str | None

_jobs: dict[str, dict[str, Any]] = {}


def _get_job(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _run_simulation(job_id: str, params: SimulationParams) -> None:
    job = _jobs[job_id]
    job["status"] = JobStatus.RUNNING
    try:
        cfg = SimulationConfig(**params.model_dump())
        micrograph, coords = generate_micrograph(cfg)
        stack = generate_particle_stack(cfg)
        job["micrograph"] = micrograph
        job["coords"] = coords
        job["stack"] = stack
        job["status"] = JobStatus.DONE
    except Exception as exc:
        job["status"] = JobStatus.FAILED
        job["error"] = str(exc)


def _run_processing(job_id: str, params: ProcessParams) -> None:
    job = _jobs[job_id]
    job["proc_status"] = JobStatus.RUNNING
    try:
        micrograph = job.get("micrograph")
        if micrograph is None:
            raise ValueError("Micrograph not available — simulation must complete first")

        sim_params: SimulationParams = job["sim_params"]
        cfg = ProcessConfig(
            low_freq=params.low_freq,
            high_freq=params.high_freq,
            apply_wiener=params.apply_wiener,
            defocus_um=sim_params.defocus_um,
            pixel_size_a=sim_params.pixel_size_a,
            voltage_kv=sim_params.voltage_kv,
            cs_mm=sim_params.cs_mm,
            snr_estimate=params.snr_estimate,
            min_sigma=params.min_sigma,
            max_sigma=params.max_sigma,
            num_sigma=params.num_sigma,
            pick_threshold=params.pick_threshold,
            min_distance=params.min_distance,
            box_size=sim_params.box_size,
            n_classes=params.n_classes,
            random_state=params.random_state,
        )
        pipeline = Pipeline(cfg)
        pipeline.run(micrograph)
        job["pipeline"] = pipeline
        job["proc_status"] = JobStatus.DONE
    except Exception as exc:
        job["proc_status"] = JobStatus.FAILED
        job["proc_error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes — simulation
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=JobResponse, status_code=202)
async def simulate(
    params: SimulationParams,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Submit a simulation job.  Returns immediately with a job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": JobStatus.PENDING,
        "sim_params": params,
        "micrograph": None,
        "coords": None,
        "stack": None,
        "pipeline": None,
        "proc_status": None,
        "error": None,
        "proc_error": None,
    }
    background_tasks.add_task(_run_simulation, job_id, params)
    return JobResponse(job_id=job_id, status=JobStatus.PENDING,
                       message="Simulation queued")


# ---------------------------------------------------------------------------
# Routes — processing
# ---------------------------------------------------------------------------

@router.post("/process", response_model=JobResponse, status_code=202)
async def process(
    params: ProcessParams,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Submit a processing job for an existing simulation job."""
    job = _get_job(params.job_id)
    if job["status"] != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Simulation not done yet (status={job['status']})",
        )
    job["proc_status"] = JobStatus.PENDING
    background_tasks.add_task(_run_processing, params.job_id, params)
    return JobResponse(job_id=params.job_id, status=JobStatus.PENDING,
                       message="Processing queued")


# ---------------------------------------------------------------------------
# Routes — status
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    """Poll simulation + processing status for a job."""
    job = _get_job(job_id)
    pipeline: Pipeline | None = job.get("pipeline")
    picks = pipeline.result.picks if (pipeline and pipeline.result) else None
    avg   = pipeline.result.class_avg if (pipeline and pipeline.result) else None

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        sim_params=job.get("sim_params"),
        n_particles_requested=job["sim_params"].n_particles if job.get("sim_params") else None,
        n_particles_placed=len(job["coords"]) if job.get("coords") is not None else None,
        n_picked=len(picks.coords) if picks else None,
        n_classes=len(avg.class_averages) if avg else None,
        error=job.get("error") or job.get("proc_error"),
    )


# ---------------------------------------------------------------------------
# Routes — results images
# ---------------------------------------------------------------------------

def _png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@router.get("/results/{job_id}/micrograph")
async def micrograph_image(job_id: str) -> StreamingResponse:
    """Return a PNG of the micrograph with picked-particle overlays."""
    job = _get_job(job_id)
    mic: np.ndarray | None = job.get("micrograph")
    if mic is None:
        raise HTTPException(status_code=404, detail="Micrograph not yet available")

    gt_coords: list = job.get("coords") or []
    pipeline: Pipeline | None = job.get("pipeline")
    picked_coords = pipeline.result.picks.coords if (pipeline and pipeline.result) else []

    fig, ax = plt.subplots(figsize=(7, 7))
    vmin, vmax = np.percentile(mic, [1, 99])
    ax.imshow(mic, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    if gt_coords:
        ys, xs = zip(*gt_coords)
        ax.scatter(xs, ys, s=40, facecolors="none", edgecolors="cyan",
                   linewidths=0.7, alpha=0.8, label="ground truth")

    if picked_coords:
        pys, pxs = zip(*picked_coords)
        ax.scatter(pxs, pys, s=40, facecolors="none", edgecolors="lime",
                   linewidths=0.7, alpha=0.8, label="picked")

    if gt_coords or picked_coords:
        ax.legend(fontsize=7, loc="upper right")

    ax.set_title(
        f"Micrograph  |  GT={len(gt_coords)}  picked={len(picked_coords)}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()

    return StreamingResponse(io.BytesIO(_png_bytes(fig)), media_type="image/png")


@router.get("/results/{job_id}/stack")
async def stack_image(job_id: str) -> StreamingResponse:
    """Return a PNG montage of the first 25 particles from the stack."""
    job = _get_job(job_id)
    stack: np.ndarray | None = job.get("stack")
    if stack is None:
        raise HTTPException(status_code=404, detail="Particle stack not yet available")

    n = min(len(stack), 25)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    for i, ax in enumerate(np.array(axes).ravel()):
        ax.axis("off")
        if i < n:
            vmin, vmax = np.percentile(stack[i], [1, 99])
            ax.imshow(stack[i], cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    fig.suptitle(f"Particle stack — first {n}/{len(stack)}", fontsize=8, y=1.01)
    fig.tight_layout()

    return StreamingResponse(io.BytesIO(_png_bytes(fig)), media_type="image/png")


@router.get("/results/{job_id}/classes")
async def classes_image(job_id: str) -> StreamingResponse:
    """Return a PNG montage of all class averages."""
    job = _get_job(job_id)
    pipeline: Pipeline | None = job.get("pipeline")
    if pipeline is None or pipeline.result is None:
        raise HTTPException(status_code=404, detail="Processing results not yet available")

    avgs = pipeline.result.class_avg.class_averages
    sizes = pipeline.result.class_avg.class_sizes
    n = len(avgs)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    for i, ax in enumerate(np.array(axes).ravel()):
        ax.axis("off")
        if i < n:
            vmin, vmax = np.percentile(avgs[i], [1, 99])
            ax.imshow(avgs[i], cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"Class {i}\nn={sizes[i]}", fontsize=7)
    fig.suptitle("Class averages", fontsize=9)
    fig.tight_layout()

    return StreamingResponse(io.BytesIO(_png_bytes(fig)), media_type="image/png")


@router.get("/results/{job_id}/process", response_model=ProcessResponse)
async def process_result(job_id: str) -> ProcessResponse:
    """Return structured JSON summary of the processing results."""
    job = _get_job(job_id)
    pipeline: Pipeline | None = job.get("pipeline")
    if pipeline is None or pipeline.result is None:
        raise HTTPException(status_code=404, detail="Processing results not yet available")

    res = pipeline.result
    picks_json = [
        PickInfo(y=y, x=x, confidence=c, sigma=s)
        for (y, x), c, s in zip(
            res.picks.coords, res.picks.confidences, res.picks.sigmas
        )
    ]
    classes_json = [
        ClassInfo(class_id=i, size=res.class_avg.class_sizes[i],
                  variance=res.class_avg.class_variances[i])
        for i in range(len(res.class_avg.class_averages))
    ]
    return ProcessResponse(
        job_id=job_id,
        n_picked=res.n_picked,
        picks=picks_json,
        classes=classes_json,
    )
