"""How close can two true particles be before non-maximum suppression merges them?

The micrograph simulator cannot answer this: its placement keeps particle boxes from overlapping, so
true centres are always at least one box (64 px) apart, far outside any suppression radius. This
benchmark bypasses placement only. Each trial builds a 128 x 128 micrograph containing exactly two
particles, using the simulator's own components (make_particle_volume, project, independent random
orientations, the same per-projection normalisation, apply_ctf, add_noise at SNR 0.1). Two particles
on 128 x 128 px is the same particle density as the benchmark micrographs (about 32 on 512 x 512),
which matters because add_noise scales the noise to the image's signal variance.

The two centres are SEPARATION px apart along a random direction through the image centre (rounded
to whole pixels). The micrograph is processed with the pipeline's own bandpass and Wiener filters and
picked with ProcessConfig defaults, varying only nms_radius. A true particle is recovered if a pick
can be matched to it one-to-one within MATCH_RADIUS px.

Nothing here is tuned; it measures the behaviour of fixed settings. Trial seeds 0-49 per separation.

    python benchmarks/nms_separation.py   ->   benchmarks/nms_separation.txt, docs/figures/nms_separation.png
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processor.filters import bandpass_filter, wiener_ctf_correction  # noqa: E402
from processor.picker import pick_particles  # noqa: E402
from processor.pipeline import ProcessConfig  # noqa: E402
from simulator.ctf import apply_ctf  # noqa: E402
from simulator.generator import SimulationConfig  # noqa: E402
from simulator.noise import add_noise  # noqa: E402
from simulator.projector import make_particle_volume, project, random_rotation_matrix  # noqa: E402

SIZE = 128
SEPARATIONS = list(range(8, 42, 2))
N_TRIALS = 50
NMS_RADII = [None, 10, 15, 20, 25, 30]
MATCH_RADIUS = 10


def pair_micrograph(sep, rng, sim=SimulationConfig(), volume=None):
    volume = make_particle_volume(sim.box_size) if volume is None else volume
    mic = np.zeros((SIZE, SIZE), np.float32)
    theta = rng.uniform(0, np.pi)
    offset = 0.5 * sep * np.array([np.sin(theta), np.cos(theta)])
    centres = [np.rint(SIZE / 2 + s * offset).astype(int) for s in (-1, 1)]
    hs = sim.box_size // 2
    for cy, cx in centres:
        proj = project(volume, random_rotation_matrix(rng))
        proj = (proj - proj.min()) / (proj.max() - proj.min())
        mic[cy - hs:cy + hs, cx - hs:cx + hs] += proj
    mic = apply_ctf(mic, sim.defocus_um, sim.pixel_size_a, sim.voltage_kv, sim.cs_mm)
    return add_noise(mic, sim.snr, sim.add_poisson, rng=rng), np.array(centres, float)


def n_recovered(picks, truth, radius=MATCH_RADIUS):
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
    return len(used_t)


def main():
    pc = ProcessConfig()
    volume = make_particle_volume(SimulationConfig().box_size)
    recall = {r: [] for r in NMS_RADII}
    picks_per_pair = {r: [] for r in NMS_RADII}
    true_sep = []
    for sep in SEPARATIONS:
        found = {r: [] for r in NMS_RADII}
        counts = {r: [] for r in NMS_RADII}
        seps = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng([sep, trial])
            mic, truth = pair_micrograph(sep, rng, volume=volume)
            seps.append(np.linalg.norm(truth[0] - truth[1]))
            f = bandpass_filter(mic, pc.low_freq, pc.high_freq, pc.pixel_size_a)
            f = wiener_ctf_correction(f, pc.defocus_um, pc.pixel_size_a, pc.voltage_kv, pc.cs_mm, pc.snr_estimate)
            for r in NMS_RADII:
                pk = pick_particles(f, pc.min_sigma, pc.max_sigma, pc.num_sigma, pc.pick_threshold,
                                    pc.min_distance, nms_radius=r)
                picks = np.array(pk.coords, float).reshape(-1, 2)
                found[r].append(n_recovered(picks, truth) / 2)
                counts[r].append(len(picks))
        true_sep.append(float(np.mean(seps)))
        for r in NMS_RADII:
            recall[r].append(float(np.mean(found[r])))
            picks_per_pair[r].append(float(np.mean(counts[r])))

    name = lambda r: "none" if r is None else f"{r} px"
    lines = [__doc__.strip().split("\n\n")[0], "",
             f"{N_TRIALS} trials per separation; recall = fraction of the two true centres recovered within "
             f"{MATCH_RADIUS} px (mean over trials); picks = mean picks per micrograph", "",
             "separation | " + " | ".join(f"NMS {name(r)}: recall, picks" for r in NMS_RADII)]
    for i, sep in enumerate(SEPARATIONS):
        lines.append(f"{sep:4d} px ({true_sep[i]:.1f}) | " + " | ".join(
            f"{recall[r][i]:.2f}, {picks_per_pair[r][i]:.1f}" for r in NMS_RADII))
    out = Path(__file__).with_suffix(".txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(recall, picks_per_pair, name)


def plot(recall, picks, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(NMS_RADII):
        style = dict(color=f"C{i}", marker="o", ms=3, lw=2.2 if r == 20 else 1.2,
                     label=f"NMS {name(r)}" + (" (default)" if r == 20 else ""))
        a.plot(SEPARATIONS, recall[r], **style)
        b.plot(SEPARATIONS, picks[r], **style)
    a.set_xlabel("separation between the two true centres [px]")
    a.set_ylabel(f"recall (fraction of the 2 true centres, {MATCH_RADIUS} px)")
    a.set_ylim(0, 1.05)
    b.set_xlabel("separation between the two true centres [px]")
    b.set_ylabel("picks per micrograph (2 true particles)")
    b.axhline(2, color="k", lw=0.8, ls="--")
    for ax in (a, b):
        ax.axvspan(15, 19, color="0.85", lw=0, zorder=0)
        ax.grid(alpha=0.3)
    a.text(15.3, 0.05, "particle size\n(15-19 px)", fontsize=8)
    a.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Two particles at controlled separation, SNR 0.1, {N_TRIALS} trials per separation", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "nms_separation.png", dpi=120)


if __name__ == "__main__":
    main()
