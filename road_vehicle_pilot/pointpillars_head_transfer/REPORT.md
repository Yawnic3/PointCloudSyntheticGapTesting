# Frozen-backbone PointPillars transfer

Outcome: head-only adaptation preserves the pretrained test accuracy but does not provide a validated improvement. Selected real, synthetic and coverage checkpoints score 5.00%, 4.99% and 5.00% 3D AP50, versus 5.00% for the unchanged pretrained control. All primary validation 3D AP50 scores remain zero.

The TRAIN-only diagnostic provides stronger evidence about the failure: the earlier selected full real fine-tune scores 65.77% on the fixed training subset but 1.35% on the real test set. Its final checkpoint scores 71.86% versus 0.65%. These are paired comparisons of the same checkpoints. The training subset covers all 15 supported training annotation IDs; these are training-fit diagnostics, not held-out scores. The result supports overfitting and a location-transfer gap in this small subset, without isolating a single cause.

The point-derived height correction worsens the selected real test AP50 to 1.09% and is not adopted. Input retention and yaw projection checks find no severe error in their tested scope. The next priority is source-only 3D object augmentation and validation across more positive locations, followed by independent confirmation on unseen real data.

Published PointPillars, 7,700 detection-head parameters trainable. All feature weights and BN buffers remain exactly unchanged. All arms use the same initialization, frames and 20-epoch update budget; source box counts differ.

Learning rate selected on real validation only: 1e-05 (real_lr1e5). Checkpoints selected on validation 3D AP50 then AP25. All four runs frozen before test inference.

| Arm | Epoch | 3D AP50 | 3D AP25 | BEV AP50 | TP / FP / FN at .25 |
|---|---:|---:|---:|---:|---|
| pretrained | 0 | 5.00% | 28.26% | 8.92% | 8 / 143 / 78 |
| real_lr1e5_best | 2 | 5.00% | 28.33% | 8.92% | 8 / 142 / 78 |
| real_lr1e5_last | 20 | 5.14% | 29.45% | 9.07% | 9 / 133 / 77 |
| real_lr1e4_best | 2 | 5.03% | 28.44% | 8.67% | 8 / 128 / 78 |
| real_lr1e4_last | 20 | 5.54% | 25.64% | 9.38% | 7 / 92 / 79 |
| synthetic_best | 2 | 4.99% | 28.26% | 9.62% | 8 / 143 / 78 |
| synthetic_last | 20 | 5.14% | 27.25% | 9.79% | 9 / 137 / 77 |
| synthetic_coverage_best | 2 | 5.00% | 28.34% | 8.92% | 8 / 143 / 78 |
| synthetic_coverage_last | 20 | 5.20% | 27.25% | 9.08% | 9 / 136 / 77 |

The two real learning-rate trials are a validation search; only the selected recipe is the matched real/synthetic comparison. The pretrained model is a fixed control; fine-tuning is not automatically considered an improvement.

27 known real test frames, 86 supported road-vehicle boxes. Diagnostic exact-cuboid AP101, not official benchmark AP. One seed/fold and no fresh blind sequences. Coverage uses real training labels as priors. No RGB input. Prior experiments and datasets preserved.

Input/checkpoint hashes rechecked after evaluation. Feature extractor hashes identical before/after every training run.

## Interpretation

The validation-selected real head-only run scores 5.00% 3D AP50, compared with 1.35% for the earlier full fine-tune and 5.00% for the unchanged pretrained model. At confidence .25 the selected run has 8 true positives, 142 false positives and 78 missed boxes.

Under the same selected learning rate and update budget, synthetic scores 4.99% and coverage-augmented synthetic 5.00%. Interpret a synthetic-versus-real gap only alongside absolute detection quality; neither a small gap nor fewer false positives establishes useful detection.

TRAIN-only retention audit: none of 175 real boxes falls below five points after the point/voxel limits. Minimum yaw-projection overlap is 95.10%. One of 557 synthetic boxes loses five-point eligibility in the deterministic voxel-cap replay. This audit does not cover every random augmentation and does not establish that all preprocessing is correct.

The real training source has only 15 supported vehicle annotation IDs from three positive locations; validation has four IDs from one location. The pretrained control has zero validation 3D AP50 yet nonzero test AP50. Validation selection is therefore fragile. The known test set remains a diagnostic, not independent confirmation.

A useful next control is a grouped validation study with more physical vehicle diversity, alongside a TRAIN-only analysis of losses on empty and positive frames. The official MMDetection3D dataset defaults to dropping empty-GT training samples, while this pilot deliberately includes 60 empty frames out of 170. That is a documented training-policy difference, not evidence by itself that negative examples caused the failure. See [official dataset source](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/datasets/det3d_dataset.py).

Verification independently checks raw/source hashes, all 20-epoch histories, all checkpoint feature/BN tensors, saved-prediction AP, and exact agreement of the pretrained control with the previous experiment. See final_verification.json.

Neither real head-only trial beats the pretrained validation control. The learning-rate choice therefore provides no validation evidence that head adaptation helps; an exact tie uses the predetermined lower-rate rule. A better known-test score alone must not be used to declare a validated replacement.

## TRAIN-only fit diagnostic

TRAIN-only fit diagnostic, never held-out performance and never used to select models

23 training frames, 28 boxes and all 15 supported training annotation IDs. Greedy coverage of all supported training annotation IDs, four spread positive frames per source location and six spread negatives. No prediction-based frame selection.

| Model | TRAIN 3D AP50 | TP / FP / FN at .25 |
|---|---:|---|
| pretrained | 23.21% | 9 / 152 / 19 |
| previous_full_real_best | 65.77% | 23 / 9 / 5 |
| previous_full_real_last | 71.86% | 23 / 6 / 5 |
| selected_head_real_best | 23.21% | 9 / 151 / 19 |
| selected_head_real_last | 23.21% | 9 / 147 / 19 |
| higher_rate_head_real_last | 23.32% | 9 / 82 / 19 |

## Secondary point-derived height correction

Fixed secondary correction: replace predicted center Z with median raw-point Z inside the predicted yaw-only XY footprint plus a source-TRAIN offset. Require five footprint points, otherwise retain original Z. No changes to XY, size, yaw, score, labels or weights. No clipping or threshold search. Offset is median over per-frame median (GT center Z minus footprint median). All full ROI points are available to this postprocessor, including points beyond the model's 160k cap.

Report all cases separately from original primary scores. Validation diagnostic is observed before new test evaluation; no test fitting or checkpoint changes. Pretrained correction uses real TRAIN calibration, so that variant is not zero-shot.

| Arm | Refined 3D AP50 | TP / FP / FN at .25 |
|---|---:|---|
| pretrained | 1.17% | 4 / 147 / 82 |
| real_lr1e5_best | 1.09% | 3 / 147 / 83 |
| real_lr1e5_last | 1.09% | 3 / 139 / 83 |
| real_lr1e4_best | 1.09% | 3 / 133 / 83 |
| real_lr1e4_last | 1.08% | 2 / 97 / 84 |
| synthetic_best | 1.18% | 4 / 147 / 82 |
| synthetic_last | 1.17% | 4 / 142 / 82 |
| synthetic_coverage_best | 1.17% | 4 / 147 / 82 |
| synthetic_coverage_last | 1.17% | 4 / 141 / 82 |
