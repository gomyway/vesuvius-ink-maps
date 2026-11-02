# vesuvius-ink-maps: registered ink maps for every public PHerc. 0800 / PHerc. 1447 segment, a low-VRAM surface_m7 runner, and an ink-inference benchmark

Open-source (MIT) tooling and outputs for the Vesuvius Challenge, built only on public data
(`s3://vesuvius-challenge-open-data`, CC BY-NC 4.0) and permissively licensed models.

What is here:

1. **Ink maps for all 21 public segments of the two Grand-Prize-eligible scrolls** (PHerc. 0800: 6, PHerc. 1447: 15),
   from reader-v2 (MIT) and Hecate (MIT), forward and reverse, plus their mean, as **2D OME-Zarr on the segment's
   tifxyz-registered canvas** and as **scroll-space VC3D overlay volumes**. Every segment passes a VC3D-loadability
   check (`scripts/check_vc3d.py`). Results table: `docs/INK_RESULTS.md`; format and usage: `docs/INK_MAPS_README.md`.
2. **A low-VRAM runner for the surface_m7 nnU-Net surface model** (`scripts/run_surface_m7_lowmem.py`): 1.6 GiB of GPU
   memory instead of 7.4 GiB for the same output (Dice 0.9999 against the original runner), so surface prediction runs on
   4 GB cards.
3. **A measured benchmark of ink inference** (`docs/BENCHMARK.md`): on upstream villa `main`, `--batch-size 8 --no-compile`
   is 2.7x faster in wall time than the published default, with maps identical to within 1/255; peak VRAM under 1 GiB.
4. **Two small villa PRs** (opened against `ScrollPrize/villa`): fail fast when the LZW TIFF codec is missing, and document the
   fast path plus the undocumented `--roi` / `--loader-threads` flags.

## Install

```bash
git clone <this repo> && cd vesuvius-ink-maps
python -m venv .venv && . .venv/bin/activate
pip install torch numpy zarr numcodecs tifffile imagecodecs opencv-python-headless scipy scikit-image boto3
# ink models
pip install "git+https://github.com/ScrollPrize/villa.git#subdirectory=vesuvius"   # upstream flat inference (vesuvius.ink_detection)
git clone https://huggingface.co/domenicor046/reader-v2 models/reader-v2             # reader-v2 checkpoints + infer_with_hecate.py (MIT)
git clone https://huggingface.co/scrollprize/hecate models/hecate                    # Hecate 9.6 um checkpoint (MIT)
# surface model (Apache-2.0) for the low-VRAM runner
git clone https://huggingface.co/scrollprize/surface_m7_nnunet models/surface_m7
pip install dynamic-network-architectures
```

Hardware used: one shared NVIDIA RTX 4090 (24 GB) and one shared RTX 5090 (32 GB). Nothing here needs more than 4 GiB
of GPU memory except the original surface_m7 runner, which is included only as the comparison baseline.

## Usage

All scripts resolve the repository root from their own location; set `VIM_ROOT` to override it and `PYTHON` to choose the
interpreter used for subprocesses (default: the running interpreter / `python`). Data, models and logs live under `data/`,
`models/` and `logs/` of that root and are not part of the repository (see Outputs manifest).

### Ink maps for one segment (GPU stage, export, check)

```bash
# 1. fetch the 21 public segments (surface volume zarr + tifxyz) listed in inventory/eligible_segments.md
scripts/fetch_segments.sh
# 2. reader-v2 + Hecate, forward and reverse (batch 8, --no-compile; Hecate bf16 at 9.6 um)
python scripts/ink_gpu.py PHerc1447 20250702235910
# 3. registered 2D OME-Zarr maps, contact sheet, scroll-space VC3D overlay
python scripts/export_ink.py PHerc1447 20250702235910
# 4. VC3D-loadability check (meta.json, level shapes, canvas registration, overlay readback at 2000 surface points)
python scripts/check_vc3d.py PHerc1447 20250702235910
# or all 21 segments, detached, with retry on OOM:
python scripts/run_ink_all.py
```

Read a map in Python:

```python
import zarr
ink = zarr.open('data/ink/PHerc1447/20250702235910/ensemble_fwd.ome.zarr/0', mode='r')[:]  # (H, W) uint8 = round(255 p)
```

Open the scroll-space overlay in VC3D: File > Open Data Catalog, add
`data/ink/PHerc1447/vc3d_overlays/PHerc1447_ink_ensemble_fwd.zarr` as a volume, select it in the Volume Overlay panel.

### Low-VRAM surface_m7

```bash
# stream a 640^3 L0 block (5 x 5 x 5 chunks of 128^3, uint8 ZYX .npy) from the public bucket
python scripts/fetch_chunk.py PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr 12416 2432 3840 5 data/sample/vol_640.npy
# original runner: ~7.4 GiB torch allocation
python scripts/run_surface_m7.py        models/surface_m7 data/sample/vol_640.npy out/surf_640_p128.npy        --patch 128
# low-VRAM runner: ~1.6 GiB torch allocation, same output (Dice 0.9999)
python scripts/run_surface_m7_lowmem.py models/surface_m7 data/sample/vol_640.npy out/surf_640_p128_lowmem.npy --patch 128
```

Both print one JSON line with `loop_s`, `Mvox_per_s`, `torch_max_allocated_MiB` and `surface_frac`.

### surface_m7 over one public segment as a VC3D overlay

```bash
# bbox of the segment's tifxyz (+32 voxels, 128-aligned) -> stream raw L0 chunks -> low-VRAM runner -> sparse scroll-space
# OME-Zarr data/ink/<scroll>/vc3d_overlays/<scroll>_surface_m7.zarr (VC3D meta.json) -> check (meta, level shapes, readback)
SURFACE_DEVICE=cuda python scripts/surface_segment_cpu.py PHerc0800 20251029010146     # or SURFACE_DEVICE=cpu (slow)
```

Done so far: 3 of 21 segments; per-segment settings and check results in `MANIFEST_surface.json` ("segment_overlays"). The first (PHerc0800 20251029010146) ran on CPU at patch 192 / step 0.75 (252 tiles in 90 min, check 11/11); GPU runs use patch 128 / step 0.5, the 1.6 GiB setting. See `docs/INK_RESULTS.md`.

### Fast ink inference on upstream villa

```bash
python -m vesuvius.ink_detection.inference.infer surface-volume.zarr reader-v2-step040000.pth out.tif \
  --overlap 0.5 --blend-mode hann --batch-size 8 --no-compile
```

## Benchmark (from docs/BENCHMARK.md)

Sample: PHerc. 1447 segment 20250702235910, 31 x 2980 x 3240 voxels at 8.64 um (7.21 cm2 bounding box). Shared cards.

| what | host | wall | loop s | cm2(bbox)/min | peak VRAM MiB |
|---|---|---|---|---|---|
| reader-v2, published default (b1, compile), warm | RTX 4090 | 17.8 s | 12.6 | 34 | 680 |
| reader-v2, published default, warm | RTX 5090 | 18.6 s | 13.6 | 32 | 858 |
| reader-v2, **b8 --no-compile** | RTX 4090 | 9.6 s | 5.6 | 77 | 1052 |
| reader-v2, **b8 --no-compile** | RTX 5090 | 8.8 s | 5.3 | 81 | 1186 |
| Hecate 9.6 um bf16 b8 | RTX 5090 | 54 s | | | 3742 |
| upstream main, published default, warm | RTX 4090 | 32.0 s | 26 | 16 | 680 |
| upstream main, b4 + compile | RTX 4090 | 61.6 s | 56 | 8 | 802 |
| upstream main, **b8 --no-compile** | RTX 4090 | 11.6-11.9 s | 8 | 53 | 980 |
| upstream main, b8 --no-compile --loader-threads 8 --chunk-cache-size 64 | RTX 4090 | 10.1 s | 7 | 61 | 980 |

| surface_m7 runner (640^3 block, patch 128) | host | torch max alloc MiB | per-process peak MiB | loop s | Dice vs original |
|---|---|---|---|---|---|
| original | RTX 5090 | 7,424 | 8,200 | 27.5 | (reference) |
| **low-VRAM** | RTX 4090 (shared, 100% util) | **1,577** | **2,524** | 85.5 | 0.9999 |

Same-card A/B (RTX 5090, 384^3 block, patch 192, 3 interleaved reps each, `docs/BENCHMARK.md`): the low-VRAM runner's loop is
3.53-3.58 s vs 2.57-3.33 s for the original (median 1.36x slower), outputs byte-identical, torch allocation 3,920 vs 4,838 MiB.
The memory saving grows with volume size (4.7x at 640^3 / patch 128) because the accumulators, not the network, move to CPU.

## Outputs manifest

`MANIFEST_ink.json`: 37,054 files, 3.02 GB, one tree sha256 per segment directory and per overlay volume (ink overlays plus the surface_m7 overlay).
`MANIFEST_surface.json`: the surface_m7 sample-block outputs with sha256. The data itself is published separately
(see the manifest for paths and sizes); it is derived from CC BY-NC 4.0 scroll data and keeps that license.

## Credits

This work builds entirely on public data, public models and public code:

- **Vesuvius Challenge data** (CC BY-NC 4.0): scans and segments of **PHerc. 0800** and **PHerc. 1447** from the public bucket
  `s3://vesuvius-challenge-open-data`, produced by the Vesuvius Challenge (Scroll Prize) and scanned/curated with
  **EduceLab**, University of Kentucky. Data terms and attribution rules: https://scrollprize.org/data. The segment ids used are
  listed in `inventory/eligible_segments.md`. Outputs derived from this data keep its CC BY-NC 4.0 terms.
- **ScrollPrize/villa** (MIT, https://github.com/ScrollPrize/villa): the `vesuvius.ink_detection` flat-inference code that
  the ink benchmark and the two PRs target. Benchmarks and PRs are against upstream `main` at commit `e89cdf3`
  (2026-11-01); the earlier re-baseline used `d57c28e`.
- **Ink models and their lineage**: the ink detectors descend from the Vesuvius Challenge 2023 Grand Prize ink-detection
  work by Youssef Nader, Luke Farritor and Julian Schilliger (TimeSformer-based ink model, MIT;
  https://github.com/younader/Vesuvius-Grandprize-Winner), carried forward in villa's `ink_detection`. We ran:
  **reader-v2** (MIT) by domenicor046, https://huggingface.co/domenicor046/reader-v2, and **Hecate** (MIT) by the
  Vesuvius Challenge team, https://huggingface.co/scrollprize/hecate (9.6 um checkpoint).
- **Surface model**: **surface_m7_nnunet** (Apache-2.0), https://huggingface.co/scrollprize/surface_m7_nnunet, the nnU-Net
  component of the 1st-place solution to the Kaggle *Vesuvius Challenge - Surface Detection* competition (see the model card
  for the solution write-up and its authors), running on nnU-Net v2 (MIC-DKFZ) and `dynamic-network-architectures`.
- **VC3D** (volume-cartographer, part of villa) defines the overlay/`meta.json` conventions that `check_vc3d.py` replays.

## Citation

If you use the maps, the runner or the benchmark, cite this repository (see `CITATION.cff`) **and** the upstream works above:
the Vesuvius Challenge / EduceLab data (per https://scrollprize.org/data), ScrollPrize/villa, reader-v2, Hecate and
surface_m7_nnunet (model cards), and the 2023 Grand Prize ink-detection authors.

## Layout

```
LICENSE                 MIT
README.md, CITATION.cff
scripts/                all code (MIT): ink pipeline, surface runners, benchmark harness, VC3D check, manifest
docs/                   BENCHMARK.md, INK_RESULTS.md, INK_MAPS_README.md, inventory tables, images
MANIFEST_ink.json, MANIFEST_surface.json
```

## Caveats

- Ink models produce letter-like artifacts; no reading is claimed from these maps. Forward and reverse directions are
  both provided because the writing side is not known for every segment.
- All timings are from shared GPUs; the card state during each run is recorded next to the numbers.
