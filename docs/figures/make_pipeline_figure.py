"""Run the simulator and processing pipeline once, at default settings, and plot each stage.

SimulationConfig(n_particles=100, seed=42) and ProcessConfig() defaults, as in the README's
Python API example (n_classes=5). Picks are matched to the known ground-truth centres by
greedy nearest-neighbour assignment within MATCH_RADIUS pixels; the counts are printed.

    python docs/figures/make_pipeline_figure.py   ->   docs/figures/pipeline_stages.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from processor.pipeline import Pipeline, ProcessConfig
from simulator.generator import SimulationConfig, generate_micrograph

MATCH_RADIUS = 10  # pixels; equals ProcessConfig.min_distance

sim_cfg = SimulationConfig(n_particles=100, seed=42)
mic, truth = generate_micrograph(sim_cfg)
proc = Pipeline(ProcessConfig(n_classes=5)).run(mic)
picks = np.array(proc.picks.coords, dtype=float).reshape(-1, 2)
truth = np.array(truth, dtype=float)

# greedy one-to-one matching, closest pairs first
d = np.linalg.norm(picks[:, None, :] - truth[None, :, :], axis=2) if len(picks) else np.zeros((0, len(truth)))
matched_p, matched_t = set(), set()
for idx in np.argsort(d, axis=None):
    i, j = np.unravel_index(idx, d.shape)
    if d[i, j] > MATCH_RADIUS:
        break
    if i not in matched_p and j not in matched_t:
        matched_p.add(i)
        matched_t.add(j)
tp = len(matched_p)
print(f"ground truth {len(truth)}, picked {len(picks)}, matched within {MATCH_RADIUS} px: {tp} "
      f"(precision {tp / max(len(picks), 1):.2f}, recall {tp / len(truth):.2f})")
print(f"class sizes {proc.class_avg.class_sizes}")


def robust(img):
    lo, hi = np.percentile(img, [1, 99])
    return dict(vmin=lo, vmax=hi, cmap="gray")


fig = plt.figure(figsize=(11, 8.2))
fig.patch.set_facecolor("white")
top, bottom = fig.subfigures(2, 1, height_ratios=[2.55, 1], hspace=0.02)
ax_raw, ax_flt = top.subplots(1, 2)
ax_raw.imshow(mic, **robust(mic))
ax_raw.set_title(f"simulated micrograph (SNR {sim_cfg.snr}, defocus {sim_cfg.defocus_um} µm)", fontsize=9)
ax_flt.imshow(proc.filtered_micrograph, **robust(proc.filtered_micrograph))
ax_flt.scatter(truth[:, 1], truth[:, 0], s=90, facecolors="none", edgecolors="#2ca02c", lw=1.1,
               label=f"ground truth ({len(truth)})")
ax_flt.scatter(picks[:, 1], picks[:, 0], s=14, marker="x", c="#d62728", lw=1.1,
               label=f"picks ({len(picks)}; {tp} within {MATCH_RADIUS} px of a true centre)")
ax_flt.set_title("bandpass + Wiener filtered, LoG picks vs. ground truth", fontsize=9)
for ax in (ax_raw, ax_flt):
    ax.set_xticks([])
    ax.set_yticks([])
top.legend(*ax_flt.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False)
top.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.07, wspace=0.04)

bottom.suptitle("k-means class averages of the picked patches (no alignment)", fontsize=9)
for k, (ax, avg) in enumerate(zip(bottom.subplots(1, len(proc.class_avg.class_averages)),
                                  proc.class_avg.class_averages)):
    ax.imshow(avg, **robust(avg))
    ax.set_title(f"class {k + 1}: {proc.class_avg.class_sizes[k]} patches", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
bottom.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.03, wspace=0.08)
out = Path(__file__).with_name("pipeline_stages.png")
fig.savefig(out, dpi=100)
print(f"wrote {out.relative_to(ROOT)}")
