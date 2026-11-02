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

<!-- vesuvius_publish:assets start -->

The ink maps are GitHub release assets of [ink-maps-v1](https://github.com/gomyway/vesuvius-ink-maps/releases/tag/ink-maps-v1): one tar per segment directory or overlay volume, paths inside as in `MANIFEST_ink.json`. Unpack with `tar -xf <asset>`; check with `sha256sum`.

| asset | bytes | sha256 |
|---|---:|---|
| [PHerc0800_20251028213516-auto_grown_20251028213516907.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251028213516-auto_grown_20251028213516907.tar) | 22,292,480 | `09a09173c7a615370bc1e54851fb50f2671ae35f63c2c8bc0897c2a67b792748` |
| [PHerc0800_20251028220042-auto_grown_20251028220042762.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251028220042-auto_grown_20251028220042762.tar) | 115,353,600 | `77eadbd70d6b5433425c41e2774f220bf59d82e6e690b9e093295ca00ef60b5a` |
| [PHerc0800_20251028220955-auto_grown_20251028220955262.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251028220955-auto_grown_20251028220955262.tar) | 15,923,200 | `458d3f5c8c129848e050e6f6a75b8caa92ead08ae468c1a0f9aeb560c4690a41` |
| [PHerc0800_20251028222030-auto_grown_20251028222030940.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251028222030-auto_grown_20251028222030940.tar) | 15,841,280 | `86d7ba44614d9da71282394d6cb19fcfa81c0aaae06575268c32cc9bb35890ca` |
| [PHerc0800_20251028225813-auto_grown_20251028225813045.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251028225813-auto_grown_20251028225813045.tar) | 13,864,960 | `02ecfd9a20d7d0263a15d1361cf9bb51bd1e23582b509108c891ce8fbe0921f6` |
| [PHerc0800_20251029010146-auto_grown_20251029010146642.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_20251029010146-auto_grown_20251029010146642.tar) | 4,812,800 | `5bb25bc6fbf97cba616cda6d6062a6d52a82a6de59c36511e9353a15d2d2ed1c` |
| [PHerc0800_vc3d_overlays_PHerc0800_ink_ensemble_fwd.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_vc3d_overlays_PHerc0800_ink_ensemble_fwd.zarr.tar) | 48,629,760 | `a6a7651f580ba372f4f35df60e91d89e6b3a5d1a2b1bacac2f7a35c0e19afc68` |
| [PHerc0800_vc3d_overlays_PHerc0800_ink_ensemble_rev.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_vc3d_overlays_PHerc0800_ink_ensemble_rev.zarr.tar) | 53,780,480 | `058d4c2778816c5c9ecfe0a64c93fdb1390b64297238a8dfbceed0d169d47a2b` |
| [PHerc0800_vc3d_overlays_PHerc0800_ink_readerv2_fwd.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_vc3d_overlays_PHerc0800_ink_readerv2_fwd.zarr.tar) | 53,882,880 | `986a4eed475c3977f5277f2edc838d34d34a41a6693e1a3b15769748106896f5` |
| [PHerc0800_vc3d_overlays_PHerc0800_surface_m7.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc0800_vc3d_overlays_PHerc0800_surface_m7.zarr.tar) | 607,508,480 | `f8db872ac3ff44d6560a28bb0d9467ea878b5eb584893b68d25aa582818c3451` |
| [PHerc1447_20250502180708-auto_grown_20250502160708188.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502180708-auto_grown_20250502160708188.tar) | 2,754,560 | `38bc6aa779ae5eecde4ba487fc3eeabacf64b9a3dd83373b3f12433a2ec0de8e` |
| [PHerc1447_20250502180748-auto_grown_20250502160748721.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502180748-auto_grown_20250502160748721.tar) | 6,256,640 | `03d414aa295b5dc4ef85536569db54619c000bfbba360edbef63cd176e6285cb` |
| [PHerc1447_20250502182142-auto_grown_20250502161324419.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502182142-auto_grown_20250502161324419.tar) | 8,939,520 | `c017997df3c0be745d95c0eb77e6034bf11a0516b2390d86c45e435f57998b2b` |
| [PHerc1447_20250502182456-auto_grown_20250502161202782.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502182456-auto_grown_20250502161202782.tar) | 10,946,560 | `ee2e3504cf840b7a991d09e2756613f19ed6f7cad1b77becc968599563052564` |
| [PHerc1447_20250502183138-auto_grown_20250502162038685.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502183138-auto_grown_20250502162038685.tar) | 161,136,640 | `7de9935bf80ed009e439bf33879a3cf9ba3dd509f4480fab19bde449c12a2927` |
| [PHerc1447_20250502183421-auto_grown_20250502161744358.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502183421-auto_grown_20250502161744358.tar) | 9,553,920 | `c76957d27dee453e5692cd1752d9acc5411a4e8c0deb29f7e58bc3ccf740a15e` |
| [PHerc1447_20250502184201-auto_grown_20250502163549332.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502184201-auto_grown_20250502163549332.tar) | 187,832,320 | `5b0f9468aad6775ea2dccf99ec045ca6331907ba8e0d4ddce4cef64c4a0e9b01` |
| [PHerc1447_20250502184658-auto_grown_20250502163923577.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502184658-auto_grown_20250502163923577.tar) | 283,832,320 | `86d871798140176705a3740ede63876704fd2fc519affa39d2d23711c439c395` |
| [PHerc1447_20250502184845-auto_grown_20250502164121265.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502184845-auto_grown_20250502164121265.tar) | 12,871,680 | `3c183252cc59bcddcd773e2879904573ac263422fc60406b98570c8b14b392ea` |
| [PHerc1447_20250502185519-auto_grown_20250502164303733.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502185519-auto_grown_20250502164303733.tar) | 9,420,800 | `7c6d622cace6f35950a36a6b715f76a836a9f12daeaf9bb6a404eef870063543` |
| [PHerc1447_20250502205333-auto_grown_20250502181030065.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250502205333-auto_grown_20250502181030065.tar) | 237,445,120 | `e3cb521d95fa8cbbde04e0c4c46a2ff9b92c6f5b26fea209b947844480911d02` |
| [PHerc1447_20250702235910-auto_grown_20250702235910292.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250702235910-auto_grown_20250702235910292.tar) | 28,968,960 | `dd0d885dbbdf5909a59df8add7938d966b04381c461b358b63ab99cba4bf099d` |
| [PHerc1447_20250703025628-auto_grown_20250703025628283.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250703025628-auto_grown_20250703025628283.tar) | 41,420,800 | `2f7d95e3226a018f2107bcb57ffb8bc9b26cd684f88681490bed15efad9e3e9d` |
| [PHerc1447_20250703034159-auto_grown_20250703034159599.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20250703034159-auto_grown_20250703034159599.tar) | 484,239,360 | `96feab901e67c9873f9ac91c6d622013409002ae5854b9891b630de2cb9f5e4a` |
| [PHerc1447_20251105093211-z_dbg_gen_00320.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_20251105093211-z_dbg_gen_00320.tar) | 53,780,480 | `30ac0eb65700bdcf63240ad0b67e37682be197bdc222a7865ffd7f54daf842f4` |
| [PHerc1447_vc3d_overlays_PHerc1447_ink_ensemble_fwd.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_vc3d_overlays_PHerc1447_ink_ensemble_fwd.zarr.tar) | 170,915,840 | `fa2646a8b1d41ff70d997a1e7ce718aed26c7fb06dd27f178fff04910abbdc77` |
| [PHerc1447_vc3d_overlays_PHerc1447_ink_ensemble_rev.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_vc3d_overlays_PHerc1447_ink_ensemble_rev.zarr.tar) | 184,524,800 | `e6e397389cfeba9ea41ad461727f7550b809c1e28c62d638ca4fa75c3dbc87b2` |
| [PHerc1447_vc3d_overlays_PHerc1447_ink_readerv2_fwd.zarr.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/PHerc1447_vc3d_overlays_PHerc1447_ink_readerv2_fwd.zarr.tar) | 206,960,640 | `2210809bbb24db6a4138f682e71856a0393cafccfa85dbc7547949be251960c8` |
| [ink_README_status.tar](https://github.com/gomyway/vesuvius-ink-maps/releases/download/ink-maps-v1/ink_README_status.tar) | 20,480 | `0f7220622039f4941ed22b31f9bbbf5bc9012a6258c46581218b150e12a353f4` |

<!-- vesuvius_publish:assets end -->

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
