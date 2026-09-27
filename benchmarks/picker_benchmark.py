"""Picker precision/recall on synthetic micrographs with known particle centres.

Compares the picker without non-maximum suppression (nms_radius=None, the behaviour before
the option existed) against the pipeline default (nms_radius=20 px, about one particle
diameter). Everything else uses ProcessConfig defaults; micrographs use SimulationConfig
defaults with n_particles=100, which jams at about 28-35 placed particles.

Seed protocol: the NMS radius was chosen on development seeds 100-109 and is justified by
the particle size (15-19 px half-maximum extent). The numbers written here come only from
held-out seeds 200-229, which were not used to choose anything, plus the README example
(seed 42) reported separately.

A pick is a true positive if it can be matched one-to-one to a true centre within
MATCH_RADIUS px (greedy, closest pairs first).

    python benchmarks/picker_benchmark.py   ->   benchmarks/picker_benchmark.txt
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processor.filters import bandpass_filter, wiener_ctf_correction  # noqa: E402
from processor.picker import pick_particles  # noqa: E402
from processor.pipeline import ProcessConfig  # noqa: E402
from simulator.generator import (  # noqa: E402
    PlacementSaturationWarning,
    SimulationConfig,
    generate_micrograph,
)

HELDOUT_SEEDS = range(200, 230)
README_SEED = 42
MATCH_RADII = (10, 5)
CONDITIONS = [
    ("SNR 0.1 (default), n=100 requested", dict(n_particles=100)),
    ("SNR 0.05, n=100 requested", dict(n_particles=100, snr=0.05)),
    ("SNR 0.2, n=100 requested", dict(n_particles=100, snr=0.2)),
    ("SNR 0.1, n=15 (sparse)", dict(n_particles=15)),
]
PC = ProcessConfig()
VARIANTS = [("no NMS", None), (f"NMS {PC.nms_radius:g} px", PC.nms_radius)]


def n_matched(picks: np.ndarray, truth: np.ndarray, radius: float) -> int:
    if len(picks) == 0:
        return 0
    d = np.linalg.norm(picks[:, None] - truth[None], axis=2)
    used_p, used_t = set(), set()
    for idx in np.argsort(d, axis=None):
        i, j = np.unravel_index(idx, d.shape)
        if d[i, j] > radius:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    return len(used_p)


def evaluate(seeds, sim_kwargs):
    rows = {name: {r: ([], []) for r in MATCH_RADII} for name, _ in VARIANTS}
    n_truth = []
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlacementSaturationWarning)
            mic, truth = generate_micrograph(SimulationConfig(seed=seed, **sim_kwargs))
        truth = np.array(truth, float)
        n_truth.append(len(truth))
        f = bandpass_filter(mic, PC.low_freq, PC.high_freq, PC.pixel_size_a)
        f = wiener_ctf_correction(f, PC.defocus_um, PC.pixel_size_a, PC.voltage_kv, PC.cs_mm,
                                  PC.snr_estimate)
        for name, nms in VARIANTS:
            pk = pick_particles(f, PC.min_sigma, PC.max_sigma, PC.num_sigma, PC.pick_threshold,
                                PC.min_distance, nms_radius=nms)
            picks = np.array(pk.coords, float).reshape(-1, 2)
            for r in MATCH_RADII:
                tp = n_matched(picks, truth, r)
                rows[name][r][0].append(tp / max(len(picks), 1))
                rows[name][r][1].append(tp / len(truth))
    return rows, n_truth


def fmt(v):
    v = np.asarray(v)
    return f"{v.mean():.3f} (sd {v.std(ddof=1):.3f})" if len(v) > 1 else f"{v[0]:.3f}"


def main():
    lines = [__doc__.strip().split("\n\n")[0], "",
             f"held-out seeds {HELDOUT_SEEDS.start}-{HELDOUT_SEEDS.stop - 1} "
             f"({len(HELDOUT_SEEDS)} micrographs per condition); mean and sd over micrographs", ""]
    for cname, kw in CONDITIONS:
        rows, n_truth = evaluate(HELDOUT_SEEDS, kw)
        lines.append(f"[{cname}]  particles per micrograph: {min(n_truth)}-{max(n_truth)}")
        for name, _ in VARIANTS:
            for r in MATCH_RADII:
                p, rc = rows[name][r]
                lines.append(f"  {name:10s} match {r:2d} px: precision {fmt(p)}  recall {fmt(rc)}")
        lines.append("")
    rows, n_truth = evaluate([README_SEED], dict(n_particles=100))
    lines.append(f"[README example, seed {README_SEED}]  particles placed: {n_truth[0]}")
    for name, _ in VARIANTS:
        p, rc = rows[name][10]
        lines.append(f"  {name:10s} match 10 px: precision {fmt(p)}  recall {fmt(rc)}")
    out = Path(__file__).with_suffix(".txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
