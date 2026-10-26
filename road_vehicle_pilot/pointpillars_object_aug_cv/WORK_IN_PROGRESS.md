# Completed experiment record

**COMPLETE:** original20 models/all45 evaluations and four visibility-repair
models/all8 evaluations finished. Both independent verifiers passed. No GPU
training or queued reporting remains active. All original data/weights retained.

Original five-location mean AP50 (%): real23.15->22.69 with augmentation;
synthetic1.31->17.53; pretrained26.32. Visibility-aware reevaluation of those
same weights: real25.21->24.70; synthetic1.35->17.90; pretrained29.56.
The visibility-aware evaluation does not retroactively repair training/selection.

Bounded repair results: Klein Flottbek realcontrol1.07->1.45%, realaugmented
4.51->5.19%; Underground remains nearzero. Two-location mean realcontrol
.53->.72%, realaugmented2.26->2.60%. Source positives change8 frames per fold;
unchanged SYN references are not a newly positive-balanced clean benchmark.

Reports: REPORT.md here and sibling pointpillars_visibility_repair_pilot/REPORT.md.
New final_verification.json files record all hash/history/AP checks. Original
comparison.png visually inspected. The next full clean fivefold benchmark
requires visibility-aware validation selection and rebalanced frame sampling;
this record does not claim that additional study was run.

The notes below are historical handoff notes, superseded by this completion.

## Latest steering and mandatory remaining work

**Newest status (supersedes counts below): original20 models/all45 test cases
finished and independent report29563, range96851 and visibility9759 completed
successfully. Original main22628 also closed successfully. Four-model repair
58137 is active: fold3 real_control and real_augmented complete; fold4
real_control was epoch3 at last check. Repair report59102 still queued. Do NOT
finish before repair and its verifier finish.**

Verified original five-location selected-checkpoint mean AP50 (%):
pretrained26.319; real_control23.149; real_augmented22.686;
synthetic_control1.308; synthetic_augmented17.532. Most SYN improvement is
calibration6.029->81.778%; pedestrian bridge stays0, other hard sites weak.
SYN augmented has4221FP/99TP atconf.25, so AP gain is not reliability.
Visibility-aware secondary mean AP50 (%), same frozen models,589->565GT:
pretrained29.559; real_control25.215; real_augmented24.699;
synthetic_control1.354; synthetic_augmented17.900.

`diagnose_object_cv_range.py` now has optional `--visibility-aware`. That mode
completed, appending a separate visible-only table and saving
range_support_visible_diagnostic.json. It verifies counts against secondary
visibility-aware metrics. Under50 raw in-cuboid returns:0TP for all trained
models (baseline1TP among20-49). These are recall atconf.25, not AP. Do not
confuse geometric in-box returns with true object-surface returns.

Originalcomparison.png was visually inspected and readable. It shows strong
location dependence. Source PDFsections21/43/46 were fully rendered/reviewed.
No original annotations were modified. Explicit100% metadata is a task-specific
ignore policy for this LiDAR-only pilot, not a claim that original annotations
are illegitimate for other modalities/tasks. Same track can be visible later;
remove hidden observations, not entire identity.

Functions store key `repairProgress` holds the compact live query for repair
epochs/test cases. Inspect sibling pointpillars_visibility_repair_pilot REPORT,
comparison, verified_summary and final_verification when done. Reporter59102
independently checks all8 new frozen weights,4 six-epoch histories, original
inputs/weights and new saved-prediction AP. Original sourcefiles and repair
trainer + evaluate_object_cv_visibility.py are frozen; do not edit them.

User said **continue** while the 20-run experiment was active. Continue the
original study AND the bounded visibility repair announced below. No permission
question is pending; all work is authorized.

The20-model training stage is complete. All40 checkpoint hashes are frozen.
Original test evaluation was42/45 cases at the latest observed status. Main
session22628 continues. Report29563, range96851 and visibility9759 are queued.

Important discovery: our geometric support gate retains24 real GT instances
explicitly marked LiDAR occlusion100% (one Bergedorf track, some other frames
of that track are visible). That yields8 phantom-positive-only TRAIN frames
per affected fold. Full documentation and metadata audits are in
`ANNOTATION_FINDINGS.md`. The public research PDF was downloaded with approved
network escalation; PDF skill read, pages21/43/46 rendered and visually reviewed.
It says full-size estimates for occluded objects and camera-only objects can
be propagated into LiDAR as occlusion100%. SemSeg is geometric in-box indices,
not independent true object-point labels. Do NOT switch all labels to visible
point bounds. Synthetic audit165frames/520evaluatedobjects found no under5
actual actor ray returns.

**Additional active bounded repair:** session58137 waits for old main
OBJECT_CV_COMPLETE then runs `../train_visibility_repair_pilot.py` into
`../visibility_repair_training.log`. Four new real models: controls/augmented
forfolds3/4, same6epochs/frames/seed/optimizer. Hidden targets/donor observations
removed, hidden-anchor classification gradients ignored with visible-positive
priority. CPU preflight passed and is cached in sibling
`pointpillars_visibility_repair_pilot/preflight.json`. Same visible track IDs
can remain as donors; exclude observations, not whole IDs. Unaugmented XYZ
exactly matches old pipeline. Real positive frames70->62 and40->32; unchanged
SYN are references, NOT a newly balanced real/SYN study. Validation/test in
these two folds have no eligible hidden targets, so old evaluation rules match.

Repair verification/report session59102 waits for VISIBILITY_REPAIR_COMPLETE,
then runs `../report_visibility_repair_pilot.py` into
`../visibility_repair_verification.log`. Do not finish until both original
and repair pipelines complete and reports are inspected. All source files
used by frozen protocols must remain unchanged; reporting/README files can
still be edited. The repair trainer source and visibility-helper hash freeze
when actual repair training begins. Its preflight source hash already frozen.

Final readouts: original REPORT/comparison/visibility-aware/range-support JSONs
and comparison.png; sibling repair REPORT/comparison/verified_summary and
final_verification. Open final reports in Codex and summarize actual paired
augmentation effects plus actual eligibility-repair effects, without claiming
the whole model/dataset problem is solved. Mention six-epoch/known-data limits.
If summarizing the research PDF in final, PDF skill requires its source citation
directive with verified absolute PDF path and purpose=source. PDF is unchanged,
so do not mark it an output artifact. Other local reports use normal file links.

Functions store key `objectCvProgress` holds a compact exec_command string
reading the latest original training/test stage and completed case counts.
Can use `text(await tools.exec_command({cmd:load('objectCvProgress'),...}))`.

## Original preparation details

User authorized source-only 3D object augmentation and validation across more
locations. Continue this active experiment to completion; do not claim results
before evaluation finishes.

- Training process: unified exec session **22628**.
- Queued verification/report process: session **29563**, waits for
  `OBJECT_CV_COMPLETE` in `../object_cv_training.log` before running
  `report_pointpillars_object_cv.py` into `../object_cv_verification.log`.
- Queued range/support diagnostic process: session **96851**, waits for
  `OBJECT_CV_REPORT_VERIFIED`, then runs `diagnose_object_cv_range.py` into
  `../object_cv_range_diagnostic.log` and appends secondary recall tables.
- Latest progress update: **19/20** training runs complete; fold4 synthetic
  augmented is the last training run. No outer tests had started at this update.
- Training started approximately 2026-10-26 00:16 UTC. It runs all 20 models
  before any outer-test inference, then freezes 40 best/final checkpoints and
  evaluates pretrained plus best/final for each arm in every fold.
- Five folds; each has two positive TRAIN locations, two positive validation
  locations and one outer-test location. Known retrospective corpus, no fresh
  blind claim. Validation caps preserve all physical annotation IDs and every
  selected negative location. All 400 real test frames appear once per arm.
- Arms: real_control, real_augmented, synthetic_control, synthetic_augmented.
  Same published PointPillars, full fine-tuning with frozen BN running stats,
  six epochs, batch1, seed23, AdamW1e-4, cosine1e-6, BF16, grad clip10.
- The six-epoch budget is explicitly a screen, not convergence. The new splits
  differ from the old single-fold trials; compare matched controls here rather
  than claiming direct improvements over old 1.35% or 5% scores.
- Source-only augmentation: TRAIN cuboid-return donors, pose/scale copy-paste,
  vehicle-collision rejection, local-ground placement, range thinning,
  half-space partial visibility, global Z +/-1m. No physical reraycast/RGB.
  Does not create independent real vehicles. Common post-operation support>=5.
- All ten source/fold replays passed. Additional independent 120-pair SAT/BEV
  collision and tilted canonicalization checks passed; max error6.43e-8m.
- `TRAIN_augmentation_example.png` was generated and visually inspected.
- `future_raw_donor_inventory.json` is a separate read-only planning audit:
  full raw TRAIN clouds contain potential outside-ROI donor annotation IDs
  (0,6,7,4,4 additional IDs across folds). They are NOT used by this experiment.
- Protocol/source/split/input hashes are frozen. Do not modify trainer, shared
  augmentation/base/model/geometry files during this run.

Relevant source scripts in the parent pilot folder:
`build_pointpillars_object_cv.py`, `lidar_object_augmentation.py`,
`audit_pointpillars_object_aug.py`, `train_pointpillars_object_cv.py`,
`report_pointpillars_object_cv.py`, `plot_object_augmentation.py`.
`diagnose_object_cv_range.py` produces secondary recall (not AP) by fixed range
and raw-return-count bins, with exact pooled TP/FP/FN agreement checks.

Before finishing: wait for training and queued reporting to complete; inspect
`comparison.json`, `validation_gated_comparison.json`, `REPORT.md`,
`final_verification.json`, and visually inspect `comparison.png`. Verify all
20 histories have six epochs and all original inputs/checkpoint hashes match.
The reporter independently recomputes all per-fold and pooled AP. Update this
handoff as completed, open REPORT.md in Codex, and give concise actual paired
real/synthetic augmentation effects and limitations. Avoid cherry-picking an
outer location or final checkpoint. No subagents or goals were created.
