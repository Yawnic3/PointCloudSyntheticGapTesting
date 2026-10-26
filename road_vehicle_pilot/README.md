# Road-vehicle RGB pilot

## Latest LiDAR road-vehicle experiments

The [five-location 3D augmentation report](pointpillars_object_aug_cv/REPORT.md)
compares matched PointPillars models with and without source-only object
augmentation. Its original frozen scores and secondary visibility-aware
evaluation are retained separately.

The [visibility-repair report](pointpillars_visibility_repair_pilot/REPORT.md)
tests removal of explicitly100%-occluded targets from LiDAR-only supervision
and donor crops, with ignored classification regions. Original annotations
remain unchanged for other tasks. New LiDAR experiments should account for
sensor-specific visibility; an in-cuboid point count alone is insufficient.
Both reports are six-epoch known-data screens, not final benchmark results.

## Original RGB pilot

This is the first end-to-end test of the real-trained versus synthetic-trained experiment. The class is **road_vehicle**, preserving the OSDaR23 category rather than asserting all real objects are passenger cars.

Two YOLO26s models start from the identical COCO-pretrained checkpoint and seed. One trains on 63 real RGB images; the other trains on 63 synthetic RGB images. Training sets also match positive-image counts. Both are evaluated on the exact same 40 real test images, from locations excluded from training and validation. There are 20 common real validation images for monitoring. A fixed epoch budget and **last.pt** determine the evaluated checkpoint, not the best test or validation score.

## Run

In PowerShell from CROSS:

```powershell
& '.\osdar23_synthetic\road_vehicle_pilot\run.ps1'
```

This audits data and runs a one-epoch smoke test, using a timestamped run name. Existing explicit run names are protected; use a different `-Tag` for a new run. The initial setup run is saved under `smoke_real` and `smoke_synthetic`.

After expanding/reviewing the datasets, a longer pilot can use:

```powershell
& '.\osdar23_synthetic\road_vehicle_pilot\run.ps1' -Epochs 30 -ImageSize 960 -Tag pilot30
```

Do not interpret a one-epoch test as evidence for or against synthetic training. Even the longer run is limited by the current single synthetic route and twelve repeated synthetic vehicles.

## Files

- `label_review.jpg`: a small real/synthetic ground-truth overlay sheet.
- `dataset_summary.json`: image, positive-image, box and sequence counts.
- `data_audit.json`: integrity and leakage checks.
- `real.yaml`, `synthetic.yaml`: single-class datasets with different training lists and identical real validation/test lists.
- `data/`: YOLO images, labels and split lists. Ground-truth class 0 is road_vehicle.
- `real_raw/`: selected official annotation/image files, catalog, source readmes and licenses. No LiDAR archives are downloaded.
- `synthetic_masks/`: re-rendered instance-color masks; fully occluded synthetic vehicles are excluded from training labels.
- `runs/<tag>_real/weights/last.pt`, `runs/<tag>_synthetic/weights/last.pt`: trained checkpoints.
- `<tag>_results.json` and `.csv`: common-real-test AP50/AP50:95 and explicit confidence-0.25, IoU-0.5 precision/recall. AP differences are **synthetic minus real**, in 0–1 units; multiply by 100 for percentage points.

## Interpretation and next steps

This is a reproducible pipeline pilot, not the official full OSDaR23 benchmark. Real frame sampling is stratified by positive/negative images and entire location groups. The test set deliberately contains 20 positive and 20 negative images; its precision is specific to that sampled distribution. Neighboring frames and repeated real/synthetic objects remain correlated. Matched frame counts do not imply equal independent objects, scene diversity, or annotation density.

Synthetic boxes enclose visible mask pixels, with at least three visible pixels and two pixels per box dimension. Real boxes retain the original annotation convention, clipped to the image. Review occlusion/truncation policies before a substantive experiment. Synthetic cars currently have limited appearance variety; real road_vehicle includes broader vehicle types. COCO initialization already contains road vehicle concepts, so the experiment measures domain fine-tuning from common pretrained weights rather than learning cars from scratch.

Before drawing conclusions: expand synthetic vehicle shapes, poses, distances, backgrounds and lighting; create independent routes; increase independent real locations/objects; align box policies; run several seeds with a fixed budget and report uncertainty over independent sequences. Keep the final real test set untouched during these choices. LiDAR training is a later separate pilot and is not implemented here.

Source data: [official OSDaR23 catalog](https://data.fid-move.de/dataset/osdar23). Framework: [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26/). CUDA wheel versions follow [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

The installed local environment uses Python 3.12, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 and Ultralytics 8.4.188. `setup.ps1` recreates the main GPU dependencies; `requirements.lock.txt` records all installed versions. Regeneration order is `scan_real.py`, Blender `render_vehicle_masks.py`, `prepare_data.py`, `audit_data.py`, then `run_pilot.py`. The fetch scripts use official byte-range archive reads and require network access. Preserve the source licenses/readmes in `real_raw` when handling the downloaded real data.
