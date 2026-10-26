# Road-vehicle pilot diagnosis

Both models learned to detect vehicles in their training images but failed to transfer that ability to the held-out real test locations. This is evidence of overfitting and a difficult distribution shift, not an entirely broken training or label-loading pipeline. It does not isolate simulator realism as the sole cause.

## Existing checkpoints on their own training images

Same 960-pixel input, confidence 0.25, IoU matching threshold 0.5:

| Checkpoint and images | Matched vehicles | Precision | Recall |
| --- | ---: | ---: | ---: |
| Real-trained, real training images | 234 / 255 | 89.0% | 91.8% |
| Synthetic-trained, synthetic training images | 88 / 107 | 98.9% | 82.2% |
| Real-trained, real validation images | 2 / 34 | 50.0% | 5.9% |
| Synthetic-trained, real validation images | 0 / 34 | 0% | 0% |
| Real-trained, real test images | 0 / 36 | 0% | 0% |
| Synthetic-trained, real test images | 0 / 36 | 0% | 0% |

These are diagnostic image-set checks. Training-image accuracy is not generalization performance. Validation is a separate 20-image sample, and test is the original 40-image sample. Existing checkpoints, datasets and primary test results were preserved.

## Label and image inspection

- Inspected original label/prediction mosaics, both training curves, training-image predictions and enlarged vehicle crops. Inspected example boxes line up with vehicles or partially visible vehicle regions; no systematic X/Y or width/height conversion reversal was apparent in those examples. This is a sampled visual review, not a full annotation-policy validation.
- The first saved test mosaic consists of negative road-vehicle images containing trains and people, so its empty prediction image by itself does not establish failure. Positive-image overlays are now saved in this directory.
- At input width 960, median real training box size is **45.2 x 21.9 px**, synthetic training is **12.75 x 10.5 px**, real validation is **19.6 x 15.8 px**, and real test is **25.1 x 9.25 px**.
- Only **0.4%** of real training boxes have a side under 8 input pixels, versus **44.9%** of synthetic training boxes and **30.6%** of real test boxes. Scale distributions differ substantially.
- Enlarged real test examples include distant vehicles and vehicle roofs behind walls, vegetation or railings. Size, visibility and context are plausible contributors to failure; these checks do not prove their individual causal effects.
- Real training boxes have 41 unique annotation UUIDs; synthetic training contains 12 visible instance IDs. These are not matched independent-object budgets. Real UUIDs across subsequences also need not represent distinct physical vehicles.
- Three selected real training boxes carry the source attribute `occlusion = 100 %`. They are retained in the existing data; future data preparation should explicitly decide whether such labels are ignored. Synthetic labels use visible-mask extents and exclude fully occluded objects. Annotation policies still need alignment.

## Training curves

Training losses fell for both models. Real-trained box loss fell from 2.56 to 1.31 and classification loss from 11.99 to 0.86. Synthetic-trained box loss fell from 2.02 to 1.02. Meanwhile validation localization losses rose and validation AP stayed low/unstable. Real validation AP50 peaked around 14.5% but ended at 5.9%; synthetic validation AP50 ended around 0.2%. These observations support overfitting. The primary fixed-budget comparison correctly used last.pt; it should not be replaced by whichever checkpoint scores best on the test set.

## Confidence check

For diagnostic purposes only, evaluated predeclared confidence thresholds 0.001, 0.01, 0.05 and 0.25. At confidence 0.001 the real-trained model matched **7/36** test vehicles but produced **889 false positives** (0.78% precision). The synthetic-trained model still matched none of the 36 at IoU 0.5. A lower threshold is not an adequate fix. Any future operating threshold must be selected using validation data, not these test results.

The original COCO checkpoint, grouping COCO car/bus/truck predictions as road vehicles, matched 2/34 validation vehicles at confidence 0.25. That is a diagnostic comparison with a broader original class head, not another matched training experiment.

## Five-image learning check

Created five enlarged crops from **real training images only**, with all intersecting labeled road vehicles retained. Trained a fresh YOLO26s for 80 tiny-data epochs with augmentation disabled; training and validation intentionally used the same five crops. At confidence floor 0.25, this reached **70.8% AP50**, **63.6% precision** and **87.5% recall** at the library-selected operating point. This shows the pipeline can learn and localize recognizable vehicles. It is not held-out accuracy, and the crops must not be added to the official test set. See `tiny/result.json`, `tiny/memorization_check`, and `tiny/predictions`.

## Recommended next experiment

1. Establish a stronger real-data baseline using additional independent training locations/vehicles and more distant, partly occluded examples. Reduce repeated neighboring frames dominating a few cars. Select learning rate, epoch budget and checkpoint policy using real validation data only.
2. Match the synthetic training population to the real training/validation task: closer vehicles with useful pixel sizes as well as distant vehicles, multiple body types, views, occlusion levels, road positions, materials, backgrounds and lighting. Use multiple independent routes. Keep the separate small/distant-object challenge visible in reporting.
3. Align visible versus amodal box conventions, truncation rules and fully occluded-object handling. Keep road_vehicle as the class unless both datasets are explicitly relabeled for passenger cars only.
4. Compare matched independent-object/scene budgets as well as image counts; the current 255 versus 107 labeled-box occurrences are not matched supervision volume.
5. Use validation for development. The existing test set has now been examined diagnostically; reserve a fresh location-disjoint final test set for the next substantive comparison. Do not create an easier test set merely to obtain a better number. Report visibility/size strata and repeat seeds.

Do not simply extend the current runs for more epochs or treat a successful five-image check as evidence that synthetic training replaces real data.

Artifacts: `diagnosis.json` contains detailed threshold and per-object results; `real_occlusion.json` summarizes source attributes; `*_label_crops.jpg` show enlarged ground truth; `*_positive_*.jpg` show **green ground truth** and **red predictions at confidence >=0.01**. Those diagnostic overlays use a lower confidence than the 0.25 comparison table.
