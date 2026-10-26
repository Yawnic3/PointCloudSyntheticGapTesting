# Frozen-backbone road-vehicle experiment

This directory preserves the head-only follow-up to `pretrained_pointpillars`.
It uses the same published PointPillars weights, real/synthetic training lists,
real validation and known real test fold. It is a LiDAR XYZ-only, single-class
fine-tuning experiment with common KITTI pretraining.

Two real head-only trials compare learning rates 0.00001 and 0.0001. The best
real-validation 3D AP50 then AP25 selects the recipe; ties prefer the lower
rate. The chosen rate is reused for unchanged and coverage-augmented synthetic
training. All four runs receive 20 epochs. Only 7,700 detection-head parameters
are trainable; all feature weights and BN statistics remain frozen.

`REPORT.md` contains completed results. `protocol.json`, histories, checkpoint
hashes and independent verification preserve the experiment. The report always
keeps primary scores separate from the secondary point-derived height rule.
The latter uses source TRAIN annotation statistics; its pretrained variant
therefore uses real training labels and is not zero-shot.

## Reproduction

From `road_vehicle_pilot`, use the existing `.venv/Scripts/python.exe`:

```powershell
python audit_pointpillars_training.py
python -c "from refine_pointpillars_height import fit; fit()"
python train_pointpillars_head.py
python report_pointpillars_head.py
python audit_head_validation.py
python plot_pointpillars_head.py
python plot_validation_height.py
python diagnose_pointpillars_fit.py
python refine_pointpillars_height.py
python refine_pointpillars_height.py --test
python verify_pointpillars_followup.py
```

Run the reporting steps in that order: the primary verifier rebuilds the main
interpretation section; later scripts append their separate diagnostic sections.
Completed training runs are retained and skipped, but evaluation is repeated
when rerunning the main script. Incomplete runs are preserved and rejected
rather than silently treated as complete. Do not edit frozen source or data
files and reuse an old protocol as if it were a new experiment.

`TRAINING_AUDIT.md` documents input retention and known policy differences.
`validation_height_failure.png` illustrates one pretrained-control validation
case with reasonable horizontal overlap but poor vertical placement. It is an
illustration, not an estimate of overall performance.
