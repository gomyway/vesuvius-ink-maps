Hosts: host-A = a shared machine with one NVIDIA RTX 4090 (24 GB); host-B = a shared machine with one RTX 5090 (32 GB). Both cards were in use by other tenants during every run; the free memory and utilisation at launch are recorded per row.

# Ink-model VRAM/throughput benchmark (measured)
Sample: PHerc1447 segment 20250702235910 surface volume, 31 x 2980 x 3240 at 8.64 um, which is 7.21 cm2 bbox and 3.79 cm2 papyrus. It is the same input on both hosts.

Each run: `scripts/measure_run.sh`, which records `/usr/bin/time -v` and the per-process VRAM peak from `nvidia-smi --query-compute-apps` every 0.5 s. Both cards are **shared**, so "other tenants" shows the card state during the run.

Loop = koine_machines "Selected N patches" to "Wrote" (`scripts/loop_time.py`).

Runs:
- host-B (RTX 5090, 32,607 MiB) runs are from 2026-10-31.
- host-A (RTX 4090, 24,564 MiB) runs are from 2026-11-01 at nice 10, with the card free memory at launch from `logs/bench_free.log`.

| run | host | card | free MiB at launch | wall | loop s | cm2(bbox)/min (loop) | per-proc peak MiB | card util mean % | card used MiB min-max |
|---|---|---|---|---|---|---|---|---|---|
| reader-v2 published cmd (b1, torch.compile), first run incl. compile | host-A | RTX 4090 | 5488 | 0:34.22 | 27.52 | 15.7 | 786 | 100 | 19076-20255 |
| reader-v2 published cmd, warm | host-B | RTX 5090 | n/a (not logged) | 0:18.59 | 13.55 | 31.9 | 858 | 85 | 8914-9777 |
| reader-v2 published cmd, warm | host-A | RTX 4090 | 5488 | 0:17.81 | 12.59 | 34.3 | 680 | 100 | 19076-19762 |
| reader-v2 b8 --no-compile | host-A | RTX 4090 | 5488 | 0:09.63 | 5.69 | 76.0 | 1052 | 100 | 19076-20134 |
| reader-v2 b8 --no-compile (repeat) | host-B | RTX 5090 | n/a (not logged) | 0:08.82 | 5.3 | 81.4 | 1186 | 88 | 10103-11294 |
| reader-v2 b8 --no-compile (repeat) | host-A | RTX 4090 | 5488 | 0:09.59 | 5.58 | 77.4 | 1052 | 100 | 19076-20134 |
| Hecate 9.6 um bf16 b8 | host-B | RTX 5090 | n/a (not logged) | 0:53.85 | n/a | n/a | 3742 | 92 | 8914-12661 |
| Hecate 9.6 um bf16 b8 | host-A | RTX 4090 | 5488 | 1:56.02 | n/a | n/a | 3608 | 100 | 19076-23819 |
| Hecate 9.6 um bf16 b8 (repeat) | host-B | RTX 5090 | n/a (not logged) | 0:54.20 | n/a | n/a | 3742 | 92 | 10103-13850 |
| Hecate 9.6 um bf16 b8 (repeat) | host-A | RTX 4090 | 5488 | 1:56.90 | n/a | n/a | 3608 | 100 | 19076-22690 |

**Findings**
- **reader-v2 fits in about 1 GiB on both cards.** The peak is 680-1,186 MiB. b8 `--no-compile` gives 76-81 cm2(bbox)/min.
- **The 4090 and the 5090 perform within 10% for reader-v2.** Both cards were shared during these runs.
- **b8 `--no-compile` beats the published default command on both cards.**
  - The loop is 2.2-2.6x faster: 5.6 vs 12.6 s on host-A, 5.3 vs 13.6 s on host-B.
  - Wall time is 1.9-2.1x faster.
  - These numbers are on the older koine_machines branch. The upstream-main re-measure (up_* runs, after upstream PR #2003) is queued.
- **Hecate bf16 b8 peaks at 3.6-3.7 GiB.**
  - The host-A wall time (116 s) is 2.2x the host-B time (54 s).
  - Inside the ink driver on host-A, Hecate took about 52 s per direction on 3-4 cm2 segments (`logs/run_ink_all.log`).
  - So the 116 s mostly reflects contention at that moment (card util column), not the 4090 itself.
  - The host-A card sat at **100% mean utilization** in every host-A run (vs 85-92% on host-B). The card held 19-24 GiB in use by other tenants' training processes.

The host-B re-measure (`scripts/bench_upstream.py` on host-B's held gate) stays queued, which is still before the 2026-11-02 02:51Z feasibility deadline.

## Upstream villa main re-baseline (vesuvius.ink_detection, d57c28e + our imagecodecs preflight), host-A RTX 4090
Same sample, same checkpoint, overlap 0.5 hann. Free memory at launch was gated at 1,600 MiB.

Upstream logs no end-of-inference line, so loop ≈ process end (start_utc.txt, 1 s resolution) minus "Selected N patches" (±1 s).

| run | wall | loop s (approx) | cm2(bbox)/min (loop) | per-proc peak MiB | card util mean % |
|---|---|---|---|---|---|
| published cmd (b1, torch.compile), first run | 0:46.49 | 39 | 11 | 680 | 99 |
| published cmd, warm | 0:32.00 | 26 | 16 | 680 | 100 |
| b4, compile (Aug workflow post) | 1:01.61 | 56 | 8 | 802 | 100 |
| b8 --no-compile | 0:11.93 | 8 | 53 | 980 | 100 |
| b8 --no-compile (repeat) | 0:11.59 | 8 | 53 | 980 | 100 |
| b8 --no-compile --loader-threads 8 --chunk-cache-size 64 | 0:10.06 | 7 | 61 | 980 | 100 |
| b1 --no-compile | 0:30.37 | 26 | 16 | 720 | 100 |

**Findings (upstream)**
- **The maps are identical across code versions and cards.**
  - Every upstream output matches the old-branch output on the host-B RTX 5090 to within 1/255 at any pixel: r = 1.00000, mean |diff| 0.004-0.013.
  - The old-branch host-B and host-A maps agree the same way.
  - So these maps are reproducible regardless of code version or card.
- **b8 `--no-compile` is still the fast path upstream.**
  - It takes 11.6-11.9 s wall vs 32.0 s for the published default command (warm): **2.7x faster**.
  - b4 with compile, the setting in the August workflow post, takes 61.6 s.
  - Adding `--loader-threads 8 --chunk-cache-size 64` (upstream #2003) gives 10.1 s, i.e. 1.15x more.
- **Upstream main was slower than the old branch on host-A for the same settings.**
  - Default: 32.0 vs 17.8 s. b8 no-compile: 11.6-11.9 vs 9.6 s.
  - Both measurements were on a card at 100% utilization from other tenants, about 15 min apart, so this is **not conclusive**.
  - It does not show #2003's 23% gain under these conditions. The host-B re-measure (queued) is the cleaner comparison.
- **Peak memory upstream is 0.66-0.96 GiB.**
- **The imagecodecs preflight patch works on the normal path.** All 7 upstream runs wrote LZW TIFFs through it.

## Low-VRAM surface_m7 runner (`scripts/run_surface_m7_lowmem.py`), host-A RTX 4090
Input: the same 640^3 L0 block of PHerc1447 as the feasibility study (640^3 block at z 12416, y 2432, x 3840 of PHerc1447 L0). Settings: patch 128, step 0.5, fp16, no mirror TTA. The card had 5,091 MiB free at launch (`logs/bench_free.log`). Run dir: `logs/runs/surface_m7_lowmem_640_p128_hostA`.

What changed from `run_surface_m7.py`:
- The input volume and the Gaussian-blending accumulators stay in CPU RAM (float32).
- Only the model and one batch of tiles are on the GPU.
- Only the surface channel is accumulated.

| runner | host / card | torch max allocated MiB | per-process peak MiB | loop s | Mvox/s | host RSS |
|---|---|---|---|---|---|---|
| original (full-block GPU accumulator) | host-B RTX 5090 | 7,424 | 8,200 | 27.5 | 9.54 | n/a |
| **low-VRAM** | host-A RTX 4090 | **1,577** | **2,524** | 85.5 | 3.07 | 8.1 GB |

**Findings**
- **GPU memory falls 4.7x by torch allocation and 3.2x by per-process peak.** surface_m7 now fits a 4 GB card, and our 8 GiB slice, with room to spare.
- **The output matches the original runner.** Against host-B's original p128 map, the low-VRAM map has Dice@0.5 = 0.9999, max |diff| = 3/255, and 98.7% of voxels identical.
- **The loop is 3.1x slower, and the comparison is confounded.**
  - The cards differ.
  - host-A ran at 100% utilization from other tenants.
  - The CPU-side accumulation is single-threaded numpy (102% CPU).
  - A same-card A/B is needed before attributing the slowdown. The original runner on host-A would give it.

## Same-card A/B: original vs low-VRAM surface_m7 runner, host-B RTX 5090, 2026-11-02 00:50Z
Same card, same input (`data/sample/vol_384.npy`, the 384^3 L0 block of PHerc1447), same settings (patch 192, step 0.5, fp16,
batch 1, no TTA), interleaved original / low-VRAM, 3 reps each, inside the held 8 GiB slice (26,252 MiB free at launch; the
card was otherwise lightly used). Driver: `scripts/ab_surface_hostB.sh`; summary rows in `logs/ab_surface.log`; per-run
`cmd.txt`, `time.log`, `gpu_trace.csv`, `gpu_apps.csv`, `measure.txt`, `stdout.log` in `logs/runs/ab_{orig,lowmem}_384_p192_r{0,1,2}`.
384^3 rather than 640^3 because the original runner peaks at 8.2 GiB per process at 640^3, above the 8 GiB slice.

| runner | rep | wall | loop s | Mvox/s | torch max alloc MiB | torch max reserved MiB | nvidia-smi per-process peak MiB | vs Oct-31 host-B original p192 map |
|---|---|---|---|---|---|---|---|---|
| original | r0 | 0:13.55 | 2.61 | 21.7 | 4,838 | 6,382 | 6,974 | identical (Dice 1.0, max diff 0) |
| low-VRAM | r0 | 0:10.41 | 3.58 | 15.8 | 3,920 | 5,518 | 6,110 | identical |
| original | r1 | 0:09.77 | 2.57 | 22.1 | 4,838 | 6,382 | 6,974 | identical |
| low-VRAM | r1 | 0:11.73 | 3.56 | 15.9 | 3,920 | 5,518 | 6,110 | identical |
| original | r2 | 0:10.70 | 3.33 | 17.0 | 4,838 | 6,382 | 6,974 | identical |
| low-VRAM | r2 | 0:10.50 | 3.53 | 16.0 | 3,920 | 5,518 | 6,110 | identical |

**Findings**
- **Speed cost of the low-VRAM runner on the same card: loop 3.53-3.58 s vs 2.57-3.33 s, i.e. 1.06-1.39x slower (median 1.36x).**
  Wall time overlaps (10.4-11.7 s vs 9.8-13.6 s) because model load dominates at this size. The 3.1x gap seen earlier across
  cards (host-A 4090 at 100% utilisation vs host-B 5090) was mostly the card and its contention, not the runner.
- **Outputs are byte-identical to the original runner's map from Oct 31 on this card** (Dice 1.0, max |diff| 0, 100% of
  voxels equal) for both runners in all 6 runs.
- **The memory saving depends on volume size, not on patch size.** At 384^3 with patch 192 the low-VRAM runner allocates
  3,920 vs 4,838 MiB (1.23x less; per-process 6,110 vs 6,974), because at patch 192 the network activations (about 3.9 GiB in
  fp16) are the floor and the 384^3 accumulators are small. At 640^3 with patch 128 the saving was 4.7x (1,577 vs 7,424 MiB).
  So for a 4 GB card use patch 128 (activation floor about 1.5 GiB) and any volume size; for an 8 GB card patch 192 works for
  any volume size with the low-VRAM runner, whereas the original runner is limited to about 384^3 at patch 192.

## surface_m7 on CPU (no GPU), host-B host, 24 threads shared, 2026-11-02
`scripts/cpu_patch_timing.py` (fp32, idle-ish box): 6.36 s per 128^3 patch (0.33 Mvox/s), 14.5 s per 192^3 patch (0.49 Mvox/s).
Real segment run (`scripts/surface_segment_cpu.py`, PHerc0800 20251029010146, 1024x896x896 block, patch 192, step 0.75, 252 tiles)
while other tenants used most of the box (about 7.5 cores available): loop 5,412 s, 0.152 Mvox/s, i.e. 20x slower than the
low-VRAM runner on the contended RTX 4090 (3.07 Mvox/s) and 100x slower than the RTX 5090 (16 Mvox/s). Output check 11/11.
