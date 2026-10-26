# Source-only 3D augmentation across five outer locations

Selected-checkpoint mean test 3D AP50 across five locations: real control 23.15%; real augmented 22.69%; synthetic control 1.31%; synthetic augmented 17.53%. Pretrained control: 26.32%.

Matched six-epoch screening trial using the same published PointPillars, optimizer, seed, per-fold source frame counts and input/evaluation rules. This is fine-tuning with common KITTI pretraining, not training from scratch. All 20 runs and 40 selected/final weights were frozen before new outer-test inference.

Each fold has two positive training locations, two positive validation locations and one positive test location. Selection uses equal-weight validation-location 3D AP50 then AP25; common negative-only validation groups are included in each location score. Validation caps retain all vehicle annotation IDs. Outer tests retain all 400 real frames once per arm and checkpoint policy.

## Matched trained-checkpoint comparison

Primary metric is equal-weight mean of the five outer-test-location AP50 values. Pooled AP is secondary; confidence counts use .25 and exact 3D IoU .5. All values are single-class road_vehicle diagnostic AP101, not official benchmark AP.

Existing GT eligibility requires the complete XY cuboid inside X[0,160), Y(-40,40) and at least five in-cuboid raw returns. Prediction centers must be inside the ROI; excluded GT is not treated as an ignore region in this diagnostic scorer. These rules are identical across all arms and unchanged from earlier pilots.

| Arm / checkpoint | Macro 3D AP50 | Pooled 3D AP50 | Pooled TP / FP / FN at .25 |
|---|---:|---:|---|
| pretrained | 26.32% | 15.03% | 113 / 2262 / 476 |
| real_control_best | 23.15% | 11.98% | 73 / 60 / 516 |
| real_control_last | 23.88% | 5.76% | 66 / 208 / 523 |
| real_augmented_best | 22.69% | 11.62% | 91 / 261 / 498 |
| real_augmented_last | 22.66% | 11.75% | 91 / 248 / 498 |
| synthetic_control_best | 1.31% | 0.27% | 27 / 1656 / 562 |
| synthetic_control_last | 0.77% | 0.31% | 18 / 1977 / 571 |
| synthetic_augmented_best | 17.53% | 10.61% | 99 / 4221 / 490 |
| synthetic_augmented_last | 14.25% | 8.88% | 88 / 3431 / 501 |

## Paired augmentation effects

| Training source | Selected-checkpoint macro change | Locations improved / tied / worsened |
|---|---:|---|
| real | -0.46 percentage points | 3 / 0 / 2 |
| synthetic | +16.22 percentage points | 4 / 1 / 0 |

## Selected-checkpoint results by held-out location

| Test location group | Pretrained | Real control | Real augmented | Synthetic control | Synthetic augmented |
|---|---:|---:|---:|---:|---:|
| 1 | 93.92% | 80.66% | 75.53% | 6.03% | 81.78% |
| 4 | 0.00% | 0.00% | 0.01% | 0.00% | 0.00% |
| 5 | 34.20% | 34.02% | 33.37% | 0.51% | 3.48% |
| 6 | 3.48% | 1.07% | 4.51% | 0.00% | 2.37% |
| 7 | 0.00% | 0.00% | 0.00% | 0.00% | 0.03% |

## Validation gate

A deployment safeguard retains the pretrained control unless fine-tuning strictly improves its two-location validation score. The table above always reports genuinely trained checkpoints. This secondary gate may retain epoch zero; it must not be described as a newly learned model in those folds.

| Arm | Gated macro AP50 | Folds retaining pretrained |
|---|---:|---:|
| real_control | 26.32% | 4 / 5 |
| real_augmented | 26.15% | 4 / 5 |
| synthetic_control | 26.32% | 5 / 5 |
| synthetic_augmented | 26.32% | 5 / 5 |

## What changed and limits

Augmented training copies measured in-cuboid returns from that source's own TRAIN data, changes pose/scale, thins returns with distance, removes a half-space for partial visibility, rejects vehicle collisions and varies global height. Background returns inside inserted boxes are removed. Donor crops can contain ground/clutter; source viewpoints and shadows are not physically reraycast. This creates training poses and contexts, not independent physical vehicles or paired RGB images. Original datasets remain unchanged.

Real and synthetic match frame counts, positive/negative frames, initialization and updates. Their box counts and vehicle diversity differ. Both arms use real TRAIN point statistics for common real-evaluation normalization; real validation annotations select checkpoints. Synthetic donor augmentation itself uses no real annotation priors.

This known corpus has prior project exposure. Six epochs, one seed, very few physical vehicles and five locations are a screening study. New splits use only two positive training locations per fold, so these scores cannot be directly compared with the earlier single-fold 1.35% or 5.00% results. A gain here requires longer matched confirmation and unseen real data before generalization claims.

Independent verification checks all data/source/list hashes, all completed histories, the common published initialization, all frozen checkpoint hashes, disjoint locations and AP recomputation from saved predictions. See final_verification.json. No outer-test scores select parameters, epochs or recipes.

## Secondary distance and point-support diagnostic

Secondary diagnostic from already-frozen selected-checkpoint predictions. Recall at confidence .25 and exact 3D IoU .5, not AP. Fixed range and raw in-cuboid-return bins. No model, threshold or recipe selection. Box instances are correlated frames, not independent vehicles.

### Horizontal distance

| Bin | GT boxes | Pretrained recall | Real control | Real augmented | Synthetic control | Synthetic augmented |
|---|---:|---:|---:|---:|---:|---:|
| <30m | 272 | 15.81% | 9.19% | 11.03% | 4.41% | 14.34% |
| 30-60m | 65 | 83.08% | 58.46% | 72.31% | 23.08% | 70.77% |
| 60-100m | 247 | 6.48% | 4.05% | 5.67% | 0.00% | 5.67% |
| 100m+ | 5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Raw returns inside the GT cuboid

| Bin | GT boxes | Pretrained recall | Real control | Real augmented | Synthetic control | Synthetic augmented |
|---|---:|---:|---:|---:|---:|---:|
| 5-19 returns | 32 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 20-49 returns | 101 | 0.99% | 0.00% | 0.00% | 0.00% | 0.00% |
| 50-199 returns | 162 | 11.73% | 4.94% | 8.02% | 0.00% | 4.94% |
| 200+ returns | 294 | 31.63% | 22.11% | 26.53% | 9.18% | 30.95% |

## Secondary visibility-aware diagnostic

Secondary post-hoc diagnostic motivated by annotation documentation and metadata, not model scores. Retain original ROI/support eligibility, but exclude GT explicitly marked LiDAR occlusion100%. Give visible-GT matches priority; otherwise ignore predictions with 3D IoU>=.5 to a fully-occluded eligible GT. Other original exclusions remain unchanged. Same frozen predictions/weights. Models were still TRAINED and SELECTED using the original eligibility; this does not fix their training targets or checkpoint selection.

Original eligible GT: 589; excluding 100%-occluded GT: 565. The 24 excluded box instances belong to one annotation track in Bergedorf. They contain 12-89 raw in-cuboid returns despite their fully-occluded annotation. Such returns do not prove that the sensor observed the vehicle itself.

| Arm / checkpoint | Visibility-aware macro AP50 | Visibility-aware pooled AP50 | TP / FP / FN at .25 |
|---|---:|---:|---|
| pretrained | 29.56% | 15.35% | 107 / 2262 / 458 |
| real_control_best | 25.21% | 11.88% | 71 / 60 / 494 |
| real_control_last | 26.51% | 5.62% | 64 / 208 / 501 |
| real_augmented_best | 24.70% | 11.54% | 87 / 261 / 478 |
| real_augmented_last | 24.87% | 11.69% | 88 / 248 / 477 |
| synthetic_control_best | 1.35% | 0.27% | 27 / 1656 / 538 |
| synthetic_control_last | 0.80% | 0.31% | 18 / 1977 / 547 |
| synthetic_augmented_best | 17.90% | 10.96% | 98 / 4221 / 467 |
| synthetic_augmented_last | 14.51% | 9.44% | 87 / 3431 / 478 |

The official research report, section2.4.3 (page21), specifies estimated true-size boxes for occluded objects. Section2.6.2 (page43) explains that camera objects invisible in LiDAR can receive projected LiDAR annotations with occlusion100%. This confirms that the raw in-cuboid point-count gate alone is insufficient to establish visibility. It does not establish that every other annotation is correct or that dimensions should be changed to visible-point extents. Source: [DZSF research report44/2023](https://edocs.tib.eu/files/e01fn23/1868657744.pdf), retained locally as osdar23_research_report.pdf; relevant pages were extracted and visually inspected.

A clean follow-up must remove fully-occluded targets from TRAIN supervision/donor crops and apply ignore handling to validation/test before selection, then rerun matched training. The tables here cannot substitute for that retraining.

## Secondary visible-only distance and point-support diagnostic

Secondary diagnostic from already-frozen selected-checkpoint predictions. Recall at confidence .25 and exact 3D IoU .5, not AP. Fixed range and raw in-cuboid-return bins. No model, threshold or recipe selection. Box instances are correlated frames, not independent vehicles.

### Horizontal distance

| Bin | GT boxes | Pretrained recall | Real control | Real augmented | Synthetic control | Synthetic augmented |
|---|---:|---:|---:|---:|---:|---:|
| <30m | 254 | 15.75% | 9.06% | 10.24% | 4.72% | 14.96% |
| 30-60m | 59 | 86.44% | 64.41% | 79.66% | 25.42% | 77.97% |
| 60-100m | 247 | 6.48% | 4.05% | 5.67% | 0.00% | 5.67% |
| 100m+ | 5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Raw returns inside the GT cuboid

| Bin | GT boxes | Pretrained recall | Real control | Real augmented | Synthetic control | Synthetic augmented |
|---|---:|---:|---:|---:|---:|---:|
| 5-19 returns | 30 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 20-49 returns | 94 | 1.06% | 0.00% | 0.00% | 0.00% | 0.00% |
| 50-199 returns | 147 | 8.84% | 4.08% | 6.12% | 0.00% | 4.76% |
| 200+ returns | 294 | 31.63% | 22.11% | 26.53% | 9.18% | 30.95% |
