# Official pretrained PointPillars transfer experiment

Published OpenMMLab Car checkpoint, all weights loaded strictly. Windows adapter learned features match archived reference code; CPU voxelization/NMS replace compiled ops. Native demo works with original and zero intensity.

All three fine-tunes finish before any test inference. Same validation AP50 then AP25 selection; additionally fixed final epoch. Same known diagnostic fold, not fresh blind evaluation.

| Arm / checkpoint | Epoch | 3D AP25 | 3D AP50 | 3D AP70 | Oriented BEV AP50 | TP / FP / FN at .25 and 3D IoU .5 |
|---|---:|---:|---:|---:|---:|---|
| zero_shot | 0 | 28.26% | 5.00% | 0.00% | 8.92% | 8 / 143 / 78 |
| real_best | 4 | 2.33% | 1.35% | 0.00% | 1.49% | 1 / 5 / 85 |
| real_last | 20 | 1.37% | 0.65% | 0.00% | 0.68% | 0 / 3 / 86 |
| synthetic_best | 10 | 1.65% | 0.93% | 0.08% | 2.10% | 10 / 492 / 76 |
| synthetic_last | 20 | 1.80% | 1.23% | 0.08% | 2.15% | 12 / 585 / 74 |
| synthetic_coverage_best | 4 | 5.63% | 0.97% | 0.00% | 1.64% | 7 / 362 / 79 |
| synthetic_coverage_last | 20 | 6.44% | 2.22% | 0.11% | 2.68% | 16 / 577 / 70 |

Same 170 source frames and update budget within fine-tuned arms. All start from the same published pretrained state. Fourth input channel is zero for both domains. Synthetic coverage uses real TRAIN-only priors and alters in-box points, not test labels or original dataset files.

Expanded range and height normalization are railway adaptations; this is not the original KITTI benchmark evaluation. 3D AP is the existing diagnostic exact-cuboid AP101 evaluator, not official OSDaR23 benchmark AP. Test comprises 27 previously exposed real frames, 86 supported vehicle boxes; one seed and one fold. Sparse and partly visible vehicle populations remain limited.

## Predeclared fixed confidence 0.10 and 0.25

Report fixed-confidence counts at 0.10 (published native score threshold) in addition to the existing 0.25 counts, identically for every baseline/checkpoint. AP and checkpoint selection unchanged; no test-based threshold selection.

| Arm / checkpoint | TP / FP / FN at .10 | Precision / recall at .10 | TP / FP / FN at .25 |
|---|---|---|---|
| zero_shot | 8 / 619 / 78 | 1.28% / 9.30% | 8 / 143 / 78 |
| real_best | 3 / 42 / 83 | 6.67% / 3.49% | 1 / 5 / 85 |
| real_last | 1 / 17 / 85 | 5.56% / 1.16% | 0 / 3 / 86 |
| synthetic_best | 10 / 546 / 76 | 1.80% / 11.63% | 10 / 492 / 76 |
| synthetic_last | 12 / 600 / 74 | 1.96% / 13.95% | 12 / 585 / 74 |
| synthetic_coverage_best | 9 / 858 / 77 | 1.04% / 10.47% | 7 / 362 / 79 |
| synthetic_coverage_last | 17 / 834 / 69 | 2.00% / 19.77% | 16 / 577 / 70 |

Final verification checks raw input hashes, shared published initialization, completed training budgets, frozen checkpoint hashes, test membership and AP recomputation from saved predictions. See `final_verification.json`. Confidence cutoffs were declared before test inference and were not selected from test performance.

## Predeclared TRAIN-only object-height alignment

Secondary object-height alignment. Translate real input Z to align its TRAIN median cuboid bottom to the source-domain TRAIN median after the original normalization; restore predicted Z by the inverse shift. No test/validation fitting. Weights and ground truth unchanged. Zero-shot uses published anchor bottom -1.78m as its reference. Primary trackbed-normalized scores remain separate.

| Arm / checkpoint | Input Z shift (m) | 3D AP25 | 3D AP50 | Oriented BEV AP50 | TP / FP / FN at .25 |
|---|---:|---:|---:|---:|---|
| zero_shot | -1.580 | 29.63% | 7.26% | 10.03% | 13 / 139 / 73 |
| real_best | -1.410 | 2.33% | 1.35% | 1.49% | 1 / 5 / 85 |
| real_last | -1.410 | 1.37% | 0.65% | 0.68% | 0 / 3 / 86 |
| synthetic_best | 0.048 | 0.00% | 0.00% | 0.00% | 0 / 679 / 86 |
| synthetic_last | 0.048 | 0.01% | 0.00% | 0.02% | 0 / 713 / 86 |
| synthetic_coverage_best | 0.048 | 0.00% | 0.00% | 0.03% | 0 / 421 / 86 |
| synthetic_coverage_last | 0.048 | 0.00% | 0.00% | 0.20% | 0 / 618 / 86 |

This is a data-informed preprocessing variant, not a calibrated physical sensor extrinsic. Synthetic models additionally use real TRAIN box-height statistics for inference normalization; this must not be described as zero-real-data transfer. Models/checkpoints are not selected from these test scores.


## Interpretation and next experiment

The published pretrained detector produces some correct 3D detections, but all
three fine-tunes score below its 5.00% primary 3D AP50. The validation-selected
real, synthetic and coverage-augmented synthetic checkpoints score 1.35%, 0.93%
and 0.97%. Coverage adds only 0.04 percentage points over unchanged synthetic.
The final coverage checkpoint reaches 2.22%, but cannot replace the selected
checkpoint based on test performance; at confidence .25 it produces 16 correct
detections and 577 false positives. This experiment has not solved transfer.

The secondary height rule improves the unfine-tuned baseline to 7.26% but reduces
synthetic checkpoint 3D AP50 to zero. A single TRAIN median height translation
does not address the synthetic transfer problem. All secondary real controls
exactly match primary metrics. Original inputs and all frozen weights remain
unchanged after these evaluations.

The next controlled experiment should preserve the pretrained baseline and
compare head-only adaptation with a smaller learning rate, using validation
for selection and a fixed budget. Separately, inspect TRAIN cuboid conventions,
partial vehicle extents and sensor visibility before regenerating synthetic
scenes. These are hypotheses to test, not established causes. Reserve new,
previously unseen real sequences for confirmation; repeated tuning against
this known test fold cannot establish generalization.
