# Source-only 3D object augmentation and broader validation

This is a single-class `road_vehicle`, LiDAR XYZ-only screening experiment.
Every model uses the same published PointPillars Car initialization and the
same six-epoch full fine-tuning recipe. Each of five outer folds compares real
and synthetic training, with and without the same augmentation package.

Each fold reserves two positive locations for validation and one for testing,
leaving two for training. Training/validation/test location groups and real
vehicle annotation IDs are disjoint within each fold. Validation retains every
supported vehicle track while capping repeated frames. All 400 real scans
appear once in outer testing per arm/policy. The corpus has prior exposure;
this is retrospective evaluation, not fresh blind testing.

The augmentation database contains only that source's TRAIN cuboid crops.
Copy/paste varies pose, context, scale, range-dependent return count, partial
visibility and global height. Vehicle boxes are kept with their returns;
overlapping existing vehicle boxes are rejected. This does not create new
independent physical cars, physically rerendered LiDAR shadows, or paired RGB.
Measured cuboid crops may include ground or clutter.

Source frame and positive/negative-frame counts match. Box counts and physical
vehicle diversity differ. No real annotation priors enter synthetic donor
augmentation. Both sources use real TRAIN point statistics for common real
evaluation normalization, and real validation annotations select checkpoints.

Primary selection is the mean AP50 of the two positive validation locations,
then mean AP25; both include common negative-only validation locations. Trained
checkpoints are selected at epochs 2/4/6. The pretrained control is reported
separately, with a secondary validation gate that retains it unless fine-tuning
strictly improves validation. All 20 runs finish and their selected/final
checkpoints are frozen before outer testing.

Primary reporting uses the equal-weight mean over five outer locations.
Pooled AP is secondary. Metrics use the existing exact-cuboid AP101 evaluator;
they are not official benchmark AP. Six epochs and one seed screen a direction;
longer matched runs and unseen data are needed for confirmation.

## Reproduction

Use the existing `road_vehicle_pilot/.venv/Scripts/python.exe` from the pilot
directory:

```powershell
python build_pointpillars_object_cv.py
python audit_pointpillars_object_aug.py
python plot_object_augmentation.py
python train_pointpillars_object_cv.py
python report_pointpillars_object_cv.py
```

Do not rebuild splits after protocol freezing. Incomplete training runs are
preserved for explicit recovery. Completed runs are skipped if training is
invoked again; test evaluation is repeated. Earlier datasets and experiments
remain unchanged. `splits_full_validation_draft.json` preserves the preparation
draft before repeated validation frames were capped; no models used that draft.
