# Ink maps on all 21 public GP-scroll segments: results
Produced 2026-11-01 on a shared NVIDIA GeForce RTX 4090 (host-A, nice 10). Driver: `scripts/run_ink_all.py`; per-segment status: `data/ink/status.json`; usage: `data/ink/README.md`.

GPU s = wall time of the GPU stage: reader-v2 fwd+rev, b8 `--no-compile`, plus Hecate bf16 fwd+rev, including model start-ups and the 9.6 um resample. It ran on a shared card, so the time includes waits for other tenants. The 20250502184201 segment also includes 27 min of waiting at the memory gate (11:59:16Z to about 12:26:16Z, in logs/host-A/run_ink_all.log).

Export s = 2D OME-Zarr maps, contact sheet and scroll-space overlay splat (CPU).

cm2 = valid surface area (tifxyz-covered canvas pixels x 8.64 um squared).

Check = `scripts/check_vc3d.py` (VC3D Volume.cpp meta.json and level-shape rules, canvas registration, overlay readback at 2,000 surface points, negative control).

| scroll | segment | canvas HxW | cm2 | GPU s | export s | check | ens fwd p99 | ens rev p99 |
|---|---|---|---|---|---|---|---|---|
| PHerc0800 | 20251028222030 | 2580x2580 | 2.36 | 203 | 47 | 37/37 | 181 | 80 |
| PHerc0800 | 20251028225813 | 2300x2260 | 1.85 | 126 | 39 | 37/37 | 133 | 107 |
| PHerc0800 | 20251028213516 | 2160x2240 | 1.73 | 120 | 41 | 37/37 | 132 | 184 |
| PHerc0800 | 20251028220042 | 2040x2000 | 1.61 | 114 | 38 | 37/37 | 183 | 164 |
| PHerc0800 | 20251028220955 | 2100x2100 | 1.54 | 112 | 33 | 37/37 | 168 | 197 |
| PHerc0800 | 20251029010146 | 980x980 | 0.37 | 48 | 12 | 37/37 | 84 | 99 |
| PHerc1447 | 20250502183421 | 3820x3860 | 5.05 | 0 | 86 | 38/38 | 51 | 43 |
| PHerc1447 | 20251105093211 | 13640x8220 | 7.81 | 476 | 156 | 38/38 | 122 | 142 |
| PHerc1447 | 20250502182142 | 3760x2560 | 3.42 | 144 | 65 | 38/38 | 45 | 42 |
| PHerc1447 | 20250502185519 | 3200x2660 | 4.43 | 147 | 65 | 38/38 | 43 | 44 |
| PHerc1447 | 20250702235910 | 2980x3240 | 4.08 | 124 | 72 | 38/38 | 167 | 146 |
| PHerc1447 | 20250703025628 | 4100x4260 | 6.83 | 395 | 123 | 38/38 | 130 | 148 |
| PHerc1447 | 20250502205333 | 3760x2580 | 4.02 | 247 | 63 | 38/38 | 46 | 50 |
| PHerc1447 | 20250502182456 | 4720x3240 | 5.41 | 168 | 76 | 38/38 | 46 | 45 |
| PHerc1447 | 20250502183138 | 3220x2120 | 2.35 | 93 | 46 | 38/38 | 50 | 48 |
| PHerc1447 | 20250502184845 | 4620x2200 | 5.72 | 174 | 79 | 38/38 | 45 | 45 |
| PHerc1447 | 20250502180748 | 4120x2240 | 2.93 | 104 | 59 | 38/38 | 48 | 42 |
| PHerc1447 | 20250703034159 | 3620x5220 | 7.65 | 446 | 129 | 38/38 | 130 | 152 |
| PHerc1447 | 20250502184658 | 4220x2860 | 5.08 | 138 | 75 | 38/38 | 44 | 45 |
| PHerc1447 | 20250502184201 | 3580x2220 | 4.21 | 1746 | 54 | 38/38 | 43 | 42 |
| PHerc1447 | 20250502180708 | 2660x3540 | 3.31 | 56 | 54 | 38/38 | 38 | 38 |

**Totals:** 21 segments, 81.8 cm2 valid surface, 5181 s GPU stage (86 min), 1412 s export (24 min). Output size 2.4G (overlays: 156M 591M).

**First Letters check** (PHerc0800 is FL-eligible): all 6 PHerc0800 contact sheets were inspected, and 2 suspicious regions were zoomed at full resolution. No rows of letterforms were found. One letter-sized ring in Hecate rev on 20251028220042 is single-model, with no CT support, so it is not a lead. See PROGRESS.md 2026-11-01 10:36-10:49Z.

## surface_m7 scroll-space overlay (added 2026-11-02): 1 of 21 segments
`scripts/surface_segment_cpu.py` runs the low-VRAM surface_m7 runner over a segment's 128-aligned bounding box and writes the
surface probability into `data/ink/<scroll>/vc3d_overlays/<scroll>_surface_m7.zarr` (sparse, 6 levels, VC3D meta.json, same
shape and chunks as the scroll volume), then checks it: VC3D meta/level-shape rules, readback at 2,000 tifxyz surface points,
and 2,000 random voxels of the same box as the control. Per-segment result: `data/ink/<scroll>/<seg>/surface_check.json`;
status: `data/ink/surface_status.json`. This run used **no GPU** (none was held): CPU, 24 threads shared with other tenants.

| scroll | segment | block (z,y,x) | Gvox | fetch | device | patch / step | tiles | loop | Mvox/s | check | surface pts: mean p / frac>=0.5 | random voxels: mean p / frac>=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PHerc0800 | 20251029010146 | 1024x896x896 at (11520, 5760, 2944) | 0.82 | 822 MB, 39.6 s (20.7 MB/s) | CPU (~7.5 of 24 cores) | 192 / 0.75 | 252 | 5,412 s | 0.152 | 11/11 PASS | 0.710 / 0.744 | 0.164 / 0.152 |

The other 20 segments were not run: at the measured CPU rate the next-smallest (1.5-1.9 cm2) would take 3-4 h each; on a held
GPU the low-VRAM runner does the same block in about 4-5 min (3 Mvox/s on the contended 4090). Logs: `logs/surface_seg_20251029010146.log`.
