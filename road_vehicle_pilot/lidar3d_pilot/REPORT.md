# Raw-XYZ 3D road-vehicle experiment

Existing retrospective grouped fold 3; all test data already diagnosed previously. Not fresh blind evaluation.

Small custom PointPillars-style CNN and local-window-attention hybrid, not official pretrained model implementations.

101 recall-level interpolated AP, exact convex cuboid intersection including source roll/pitch. Experimental evaluator, not official OSDaR23 benchmark.

| Model | Training source | Selected epoch | 3D AP25 | 3D AP50 | 3D AP70 | Oriented BEV AP50 | TP / FP / FN at .25, 3D IoU .5 |
|---|---|---:|---:|---:|---:|---:|---|
| pillar_cnn | real | 10 | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 / 86 |
| pillar_cnn | synthetic | 35 | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 16 / 86 |
| pillar_transformer | real | 20 | 0.31% | 0.01% | 0.00% | 0.14% | 0 / 1 / 86 |
| pillar_transformer | synthetic | 35 | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 64 / 86 |

Same 170 training frames (110 positive, 60 negative), seed, epochs, optimizer and augmentation policy within each pair. Same initial parameter values for real/synthetic models of each architecture. No pretrained weights. More than one architecture is tested; this is exploratory development, not a tuned official model benchmark.

Predictions are seven-parameter yaw boxes; source ground truth retains full quaternion rotation for exact overlap. Training approximates roll/pitch with yaw-only boxes. Reference subset fit and analytic geometry checks precede training. These 3D AP numbers are not directly comparable to earlier axis-aligned YOLO BEV AP.

## Predeclared secondary comparison: fixed epoch 60

Secondary evaluation of last.pt at fixed epoch 60 for all four runs; primary best-validation comparison unchanged. Declared before primary test inference.

| Model | Training source | 3D AP25 | 3D AP50 | 3D AP70 | Oriented BEV AP50 | TP / FP / FN at .25, 3D IoU .5 |
|---|---|---:|---:|---:|---:|---|
| pillar_cnn | real | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 / 86 |
| pillar_cnn | synthetic | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 17 / 86 |
| pillar_transformer | real | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 / 86 |
| pillar_transformer | synthetic | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 82 / 86 |

## Predeclared secondary check: at least five points in predicted cuboid

Secondary geometry support check: retain only predicted cuboids with >=5 points from the same full feature ROI cloud. Mirrors GT minimum-support eligibility, not a tuned confidence threshold. Apply identically to every model/source and both checkpoint policies; report unfiltered primary results separately.

| Checkpoint policy | Model | Source | 3D AP50 | TP / FP / FN at .25 | Predictions retained |
|---|---|---|---:|---|---:|
| selected | pillar_cnn | real | 0.00% | 0 / 0 / 86 | 1082 / 1705 |
| selected | pillar_cnn | synthetic | 0.00% | 0 / 15 / 86 | 1558 / 1761 |
| selected | pillar_transformer | real | 0.01% | 0 / 1 / 86 | 1454 / 2257 |
| selected | pillar_transformer | synthetic | 0.00% | 0 / 58 / 86 | 1439 / 1593 |
| epoch60 | pillar_cnn | real | 0.00% | 0 / 0 / 86 | 490 / 1516 |
| epoch60 | pillar_cnn | synthetic | 0.00% | 0 / 17 / 86 | 1618 / 1785 |
| epoch60 | pillar_transformer | real | 0.00% | 0 / 0 / 86 | 914 / 2149 |
| epoch60 | pillar_transformer | synthetic | 0.00% | 0 / 77 / 86 | 1499 / 1637 |

This check changes postprocessing, not the trained model or its confidence scores. It cannot recover detections that the model never proposed. Unfiltered selected-checkpoint results remain primary. All test data were previously exposed in the project; this is exploratory development.


## Interpretation and verification

These small models trained from scratch do not generalize reliably to the existing test location. At confidence 0.25 and 3D IoU 0.5, all selected and fixed-epoch models miss all 86 vehicles. The predeclared five-point support check removes some false positives but does not recover any true positives. The tiny real-trained transformer AP comes from low-confidence detections; it is not a useful operating point. This does not establish that official pretrained PointPillars or transformer models would fail.

A separate descriptive check found median canonical box widths of 2.00 m in real training, 2.33 m in synthetic training and 1.10 m in test; median lengths are 3.70, 5.05 and 3.20 m respectively. The synthetic source includes a broader vehicle-type mixture, so its aggregate median is not a car-only comparison. Size, visibility and population differences merit further investigation; these descriptive statistics do not prove annotation errors or isolate the cause of failure. Test labels were not changed.

The strongest next direction is a supported pretrained 3D detector and better independent vehicle/visibility coverage, with training-only augmentation and validation-based selection. More small randomly initialized model swaps alone have not solved this pilot. A fresh external holdout is still unavailable.

Final verification confirms every raw input hash, identical starting weights within each architecture pair, all four completed 60-epoch runs, unchanged selected/final weights and identical test membership. See `final_verification.json`.
