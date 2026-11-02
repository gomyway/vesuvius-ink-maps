# Ink maps for every public Grand-Prize-scroll segment (PHerc0800, PHerc1447)
reader-v2 (MIT, huggingface.co/domenicor046/reader-v2, step 40000) and Hecate 9.6 um (MIT, huggingface.co/scrollprize/hecate)
ink probability maps, plus their plain mean ("ensemble"), for the 21 public segments of the two Grand-Prize-eligible scrolls
that have segments (PHerc0800: 6, PHerc1447: 15). Each map is computed in both surface-normal directions (fwd, rev).
Data: Vesuvius Challenge open data, CC BY-NC 4.0. Code: MIT (scripts/ in the workspace).

## Layout
```
<scroll>/<segment>/
  readerv2_{fwd,rev}.ome.zarr  hecate_{fwd,rev}.ome.zarr  ensemble_{fwd,rev}.ome.zarr
      2D OME-Zarr (y, x), uint8 = round(255 * p), 5 levels (2x each), 8.64 um pixels at level 0
  contact.png      CT mid layer | reader-v2 | Hecate | ensemble, one row per direction, 1 cm scale bar
  ink_meta.json    registration, per-map statistics, timings
  vc3d_check.json  loadability + registration self-test results
<scroll>/vc3d_overlays/<scroll>_ink_{ensemble_fwd,ensemble_rev,readerv2_fwd}.zarr
      sparse uint8 OME-Zarr in the scroll's voxel space (same shape and 128^3 chunks as the scroll volume,
      6 levels, meta.json), with every segment's map value written at its surface position
<scroll>/vc3d_overlays/<scroll>_surface_m7.zarr
      surface_m7 surface probability (uint8) inside the bounding boxes of the processed segments (see
      surface_status.json; 1 of 21 so far), same layout as the ink overlays
<scroll>/<segment>/surface_check.json   VC3D + readback check of the surface overlay for that segment
status.json        per-segment pipeline status (ink)
surface_status.json  per-segment surface_m7 status
```

## Registration
- **2D maps.** Each map shares the canvas of the segment's published surface volume. That canvas is the segment's tifxyz grid divided by the grid `scale` in its `meta.json` (0.05 for these segments).
  - Map pixel (row i, col j) sits at tifxyz grid position (i * scale_y, j * scale_x).
  - The level-0 voxel position is the bilinear interpolation of `x.tif`, `y.tif` and `z.tif` at that grid position.
  - Example: PHerc0800 `20251028222030` has a 129 x 129 tifxyz grid and a 2580 x 2580 map.
- **Overlays.** Each pixel's value is written at floor(z, y, x) of its surface point, dilated by one voxel in +z, +y and +x. This sets all 8 trilinear corners, so a renderer that samples at the surface point reads the map value.
  - Overlapping segments are merged by maximum.
  - Levels 1-5 are built by 2x max pooling.
  - Everything off the surface is 0, and those chunks are absent.

## Use
- **VC3D.** Open the scroll from File > Open Data Catalog. Add `<scroll>/vc3d_overlays/<scroll>_ink_ensemble_fwd.zarr` as a volume. Select it as the overlay volume in the Volume Overlay panel, with colormap, opacity and threshold.
  - The overlay lines up with the scroll volume because the dimensions are identical. The checker verifies this the way VC3D's `Volume.cpp` does: `meta.json` fields, and level shapes of ceil(dim / 2^level).
- **Python.**
```python
import zarr
ink = zarr.open('PHerc1447/<segment>/ensemble_fwd.ome.zarr/0', mode='r')[:]   # (H, W) uint8, same canvas as the surface volume
```
- **napari.** Run `napari --plugin napari-ome-zarr PHerc1447/<segment>/ensemble_fwd.ome.zarr`.

## How it was made
- Script order: `scripts/ink_gpu.py`, then `scripts/export_ink.py`, then `scripts/check_vc3d.py`. `scripts/run_ink_all.py` drives all three.
- **reader-v2.** koine_machines inference, overlap 0.5, hann blending, batch 8, `--no-compile`.
- **Hecate.** bf16, batch 8. The input is resampled to 9.6 um, as its model card requires, and the output is resized back to the native canvas, using the same steps as reader-v2's `infer_with_hecate.py`.
- Each GPU run's command, wall time, per-process VRAM and card trace are in `logs/runs/ink_*`.

## Caveats
- Ink models can produce letter-like artifacts, so no reading is claimed from these maps.
- The forward and reverse directions are both provided, because which side faces the writing is not known for every segment.
