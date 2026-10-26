# Broader retrospective LiDAR benchmark

This replaces the one-positive-location pilot comparison with five outer folds spanning all five OSDaR23 location groups that contain road-vehicle cuboids wholly inside the current ROI. It reuses the 400 already-exported real scans and preserves every previous dataset, checkpoint and result.

**This is not a fresh blind benchmark.** Every positive location has prior project exposure. The complete local annotation inventory contains only five positive location groups in the current ROI. Three disjoint splits with multiple positive locations each would require at least six. Additional unseen sequences are needed for independent confirmation; reshuffling known data cannot create them.

## What changed and why

- Entire location groups and their vehicle annotation IDs stay in one split within each fold. Training uses three positive groups, validation one and test one. Rotating the outer test reports performance across five locations instead of a single test location.
- The primary score is an equal-weight average across outer locations. Pooled out-of-fold AP is secondary because large sequences would otherwise dominate.
- Training selects at most 40 positive frames per location and at most 60 negative frames, retaining vehicle-ID and negative-location diversity. This limits repeated-frame dominance; it does not create additional independent vehicles.
- The expanded 320-scene synthetic pool samples both sides, four distance bands, headings, elevations, beam densities and range-dependent point dropout. It uses procedural ray directions and a procedural pose grid rather than empirical real ray templates or real placement records. Existing eight vehicle families and background recipes are reused.
- Each real/synthetic pair matches total, positive and negative training-frame counts, as well as YOLO26s initialization, seed, optimizer, preprocessing and 30 epochs. Object counts and diversity remain different and are recorded.
- All ten models train before any new outer-test scores are inspected. Every checkpoint is chosen by the same validation AP50:95 policy and then frozen.

Each fold still has only one positive validation/test location. The benefit is location-wise reporting and aggregation over all five folds; it is not a claim that each split now has multiple positive locations. This remains a single-class, LiDAR-only, axis-aligned BEV experiment, not full 3D AP or all-class evaluation.

## Files

- `benchmark_protocol.json`: annotation inventory, split assignments, counts, checksums and limits.
- `fold_0` through `fold_4`: real/synthetic training lists and shared validation/test configurations.
- `expanded`: new clouds, metadata, exported BEV images, labels and eight Blender scenes.
- `expanded_audit.json`: geometry, object-return alignment, family/position coverage, matched frame counts and input hashes.
- `runs/frozen_protocol.json`: training and evaluation policy frozen before new model training.
- `runs/selected_checkpoints.json`: all selected checkpoint hashes frozen before outer-test evaluation.
- `runs/REPORT.md`: completed retrospective comparison, when available.

## Reproduction

Use the existing `road_vehicle_pilot/.venv` Python and run the following scripts from the pilot folder:

1. `python build_grouped_benchmark.py`
2. Blender background: `blender --background --python generate_expanded_lidar.py`
3. `python audit_expanded_lidar.py`
4. `python train_grouped_lidar.py`

Frozen files and hashes protect against accidental protocol changes. Existing partial training runs are not silently resumed as completed experiments. Earlier draft protocols are retained separately; they were revised before any model training or new outer-test evaluation.
