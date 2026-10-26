# Fully-occluded TRAIN-target repair pilot

Repair only explicit100%-occluded TRAIN supervision/donors; ignore classification anchors centered in hidden XY footprints, with visible-positive priority. Same data frames, optimizer, seeds, initialization, augmentation parameters and validation/test eligibility for folds3/4. Eight former positive frames per fold become background-only; frames are retained to isolate target repair, so unchanged synthetic references no longer match visible-positive frame counts. Four new real models; synthetic/pretrained references reused because their TRAIN/VAL/test eligible annotations are unaffected. This is a real-label-repair pilot, not a newly balanced real/synthetic benchmark or blind confirmation.

Four new real-data runs isolate label/donor repair under the same architecture, initial weights, six epochs, frame lists, seed, augmentation package and optimizer. These two folds' eligible validation and test annotations have no occlusion100% targets, so their evaluation rule is unchanged. Original data/weights remain unchanged.

## Selected-checkpoint comparison on the two affected folds

| Held-out location | Original real control | Repaired real control | Original real augmented | Repaired real augmented | Unchanged synthetic control reference | Unchanged synthetic augmented reference |
|---|---:|---:|---:|---:|---:|---:|
| Klein Flottbek | 1.07% | 1.45% | 4.51% | 5.19% | 0.00% | 2.37% |
| Underground approach | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.03% |

## Repair effects within real training

| Variant | Original two-location mean AP50 | Repaired two-location mean AP50 | Change |
|---|---:|---:|---:|
| real_control | 0.53% | 0.72% | +0.19 percentage points |
| real_augmented | 2.26% | 2.60% | +0.34 percentage points |

## Selected versus final repaired checkpoints

| Checkpoint | Two-location mean AP50 | Pooled AP50 | TP / FP / FN at .25 |
|---|---:|---:|---|
| real_control_best | 0.72% | 0.92% | 2 / 10 / 119 |
| real_control_last | 0.72% | 0.93% | 2 / 10 / 119 |
| real_augmented_best | 2.60% | 3.70% | 7 / 93 / 114 |
| real_augmented_last | 1.72% | 2.27% | 5 / 67 / 116 |

The eligibility repair removes 24 fully-occluded training annotations per fold and excludes those observations from donor crops. The same physical annotation ID can remain in the donor database in frames where it is visible. Eight old positive frames become background-only; no source frame is removed or replaced. The preflight verifies unaugmented XYZ equality, removal of only hidden targets, zero classification gradients for hidden-region anchors and visible-positive priority.

This is a known-data, two-location, one-seed repair pilot; it does not replace the original five-location comparison or establish a newly balanced clean real/synthetic benchmark. Synthetic references retain the original positive-frame mix. A complete clean five-fold benchmark requires visibility-aware validation selection in the other folds as well. Accuracy gains or losses here must be reported as measured; fixing eligibility alone need not solve far-range/occluded-vehicle detection.

The source annotation policy is documented in the retained DZSF report44/2023, sections2.4.3 and2.6.2: estimated full-size boxes are used for occluded objects, and LiDAR-invisible camera objects can receive LiDAR boxes marked100% occluded. Source: [public research report](https://edocs.tib.eu/files/e01fn23/1868657744.pdf). The original annotations are retained for other tasks; the repair is in the derived LiDAR-only loader and loss, not an in-place alteration of the dataset.

See preflight.json, protocol.json, verified_summary.json and final_verification.json for the frozen configuration, tests and independently recomputed results.
