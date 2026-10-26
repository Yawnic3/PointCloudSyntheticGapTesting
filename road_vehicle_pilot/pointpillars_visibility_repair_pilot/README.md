# Fully-occluded target repair pilot

This is a bounded follow-up to the original five-location augmentation study.
The old five-point geometric eligibility gate retained24 LiDAR annotations
explicitly marked100% occluded. Their in-cuboid points do not prove vehicle
returns. Original annotations remain unchanged for other tasks.

Four real-data models rerun folds3/4, whose TRAIN data contains these targets
but whose eligible validation/test labels do not. The loader excludes hidden
positive targets and hidden donor observations. A vehicle annotation ID can
remain available in frames where it is visible. Classification anchors inside
hidden XY footprints are ignored unless they are visible-positive anchors.

Architecture, common KITTI initialization, six epochs, frame lists, seed,
optimizer, augmentation settings and evaluation rules match the original
models. The same eight frames per fold become background-only after repair;
they are retained rather than replaced. This isolates real target repair.
Unchanged synthetic/pretrained models are references, not a newly balanced
real/synthetic benchmark after the visible-positive frame counts change.

The preflight checks exact unaugmented XYZ equality, target removal, donor
observation isolation and classification gradients. All four new runs freeze
before their own tests. This remains a known-data, two-location, single-seed
screen, not blind confirmation or a complete visibility-clean five-fold study.

## Reproduction

From `road_vehicle_pilot`, use the existing `.venv/Scripts/python.exe`:

```powershell
python train_visibility_repair_pilot.py --prepare-only
python train_visibility_repair_pilot.py
python report_visibility_repair_pilot.py
```

The original study must finish first so unchanged reference predictions and
frozen checkpoint hashes are available. Completed repair runs are preserved
and skipped; incomplete runs require explicit recovery. Do not alter frozen
sources and reuse an old protocol as a new experiment.
