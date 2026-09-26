"""Plot the CTF and the Wiener correction filter used by the pipeline, at default settings.

Uses simulator.ctf.compute_ctf_2d along one frequency axis, with the defaults of
SimulationConfig and ProcessConfig (300 kV, Cs 2 mm, defocus 2 um, 2 A/pixel, Q = 0.07,
snr_estimate 0.1, bandpass 0.02 to 0.25 1/A). W(k) = CTF / (CTF^2 + 1/SNR), as in
processor.filters.wiener_ctf_correction.

    python docs/figures/make_ctf_figure.py   ->   docs/figures/ctf_wiener.svg
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from processor.pipeline import ProcessConfig
from simulator.ctf import compute_ctf_2d
from simulator.generator import SimulationConfig

sim, proc = SimulationConfig(), ProcessConfig()
n = 2048
ctf_line = compute_ctf_2d((1, n), sim.defocus_um, sim.pixel_size_a, sim.voltage_kv, sim.cs_mm)[0]
k = np.fft.fftfreq(n) / sim.pixel_size_a
keep = k >= 0
k, ctf = k[keep], ctf_line[keep].astype(float)

plt.rcParams.update({"font.size": 9, "svg.fonttype": "path", "svg.hashsalt": "fig"})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
fig.patch.set_facecolor("white")
for ax in (a1, a2):
    ax.axvspan(proc.low_freq, proc.high_freq, color="#e8f0e3", zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.grid(alpha=0.3)

a1.plot(k, ctf, color="#1f5fa8", lw=1.1)
a1.set_ylabel("CTF(k)")
a1.set_ylim(-1.05, 1.05)
a1.set_title(f"CTF at {sim.voltage_kv:g} kV, Cs {sim.cs_mm:g} mm, defocus {sim.defocus_um:g} µm, "
             f"{sim.pixel_size_a:g} Å/pixel (Nyquist {1 / (2 * sim.pixel_size_a):g} Å⁻¹)", fontsize=9)
a1.text(proc.low_freq + 0.003, -0.93, f"bandpass {proc.low_freq:g} to {proc.high_freq:g} Å⁻¹",
        fontsize=8, color="#3f6b2f")

for snr, style in ((proc.snr_estimate, "-"), (1.0, "--"), (10.0, ":")):
    w = ctf / (ctf**2 + 1.0 / snr)
    label = f"SNR estimate {snr:g}" + (" (pipeline default)" if snr == proc.snr_estimate else "")
    a2.plot(k, w, style, lw=1.1, label=label)
a2.set_ylabel("W(k) = CTF / (CTF² + 1/SNR)")
a2.set_xlabel("spatial frequency k [Å⁻¹]")
a2.legend(fontsize=8, loc="lower right")
a2.set_xlim(0, k.max())
fig.tight_layout()
out = Path(__file__).with_name("ctf_wiener.svg")
fig.savefig(out, facecolor="white", metadata={"Date": None})  # deterministic output
if len(sys.argv) > 1:  # optional raster preview path
    fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print(f"wrote {out.relative_to(ROOT)}")
