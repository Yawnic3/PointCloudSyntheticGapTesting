# Published pretrained PointPillars: railway transfer

This experiment uses the **published MMDetection3D PointPillars Car checkpoint**, not the earlier small randomly initialized models. The detector has 4,814,868 learned parameters. Every checkpoint tensor loads strictly, and the published SHA256 filename prefix is verified.

Official sources and license are archived under `official/` at tag `v1.4.0`; the download URLs and complete hashes are recorded in `download_manifest.json`. The checkpoint is listed in the [official model catalogue](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/pointpillars/metafile.yml). Source code is Apache-2.0; see `official/LICENSE`.

## Windows runtime and verification

WSL, Docker and the required native build tools are unavailable locally. The adapter reconstructs the same PFN, SECOND backbone, SECONDFPN and anchor heads using PyTorch, replacing compiled voxelization with NumPy and rotated NMS with the existing CPU geometry implementation. All pretrained weights and batch-normalization buffers are retained.

`reference_parity.json` compares intermediate outputs against the original archived forward methods with equivalent framework layer builders. Pillar, backbone and neck differences are zero on the supplied native demo. Anchor differences are below 0.01 mm. The bottom-Z/center-Z box conversion is verified. This verifies learned-layer equivalence; it is not a full official KITTI benchmark run or proof that every compiled voxelization/NMS edge case is bit-identical.

The official KITTI demonstration produces ten car detections, and also detects ten cars with intensity set to zero. This is a sanity check, not railway accuracy. The transfer preflight fits all three boxes of a training-only example at 3D IoU 0.5.

## Railway adaptations

- Detection remains one `road_vehicle` class, mapped to the pretrained Car head. Fine-tuning includes all annotated road vehicles.
- Inputs contain XYZ plus a fourth channel fixed to zero in both domains. No RGB, annotation IDs or sensor IDs enter the model.
- XY pillars retain the published 0.16 m resolution. The fully convolutional map extends to 160 m forward and padded Y +/-40.32 m; predictions are restricted to the existing +/-40 m evaluation region.
- Pillar vertical support expands to [-5,8] m while preserving the pretrained vertical feature-center reference at -1 m. This is an adaptation, not the original native input configuration.
- Ground offsets are fitted separately from real TRAIN and synthetic TRAIN trackbed statistics, aligning the typical ground level to -1.6 m. Validation/test use the fixed real-training offset. Elevated terrain and assumed ground height remain limitations.
- Assignment retains the published nearest-BEV IoU matching approach, with forced best matches allowed down to IoU 0.1 to supervise narrow and large road vehicles. Native Car anchors otherwise leave some of these unmatched.
- Railway postprocessing uses the same fixed settings for all arms: score floor .001, top 300 candidates, rotated BEV NMS .1, at most 100 detections. This differs from the native KITTI demo settings and is documented in code.

## Matched arms and why

All fine-tunes start from the same published weights and use the same 170 training frames, batch one, 20 epochs, AdamW learning rate .0001 and seed 23. Pretrained BN running statistics are frozen; affine parameters remain trainable. The arms are real data, unchanged synthetic data, and synthetic data with TRAIN-informed coverage augmentation. The unmodified pretrained model is also evaluated as a zero-shot baseline after training finishes.

The coverage arm draws box sizes from real TRAIN records, balances nonempty width bands, rescales in-box points and boxes together while preserving the bottom, and randomly thins object returns. It broadens narrow/partial and sparse examples without changing test labels. It is a point-level augmentation, **not a new physically raycast scene**, and does not have matching RGB renders. All original source files are retained. It uses real training annotations as priors and must not be described as zero-real-data generation.

The training-only replay retains 557 boxes, shifts median width from 2.33 m to 1.80 m, and increases the under-50-point fraction from approximately 9% to 37%. These are replay coverage statistics, not performance improvements.

## Evaluation

The existing retrospective grouped fold 3 is reused: 39 validation frames and 27 test frames with 86 supported test boxes. All source locations were previously exposed during the project, so this is **not fresh blind evaluation**. All three fine-tunes complete and selected/final checkpoints are frozen before test inference.

The primary selection uses real-validation exact 3D AP50, then AP25. Final-epoch results are reported separately for every arm. Metrics use the existing exact-cuboid AP101 diagnostic evaluator, not official OSDaR23 benchmark AP. One seed, one fold, few physical vehicles, different supported object counts and expanded input geometry limit conclusions.

## Reproduction

From `road_vehicle_pilot` with the existing `.venv`:

1. `python fetch_pretrained_pointpillars.py`
2. `python official_pointpillars_windows.py`
3. `python verify_official_pointpillars.py`
4. `python train_pretrained_pointpillars.py --prepare-only`
5. `python verify_pretrained_transfer.py`
6. `python train_pretrained_pointpillars.py`

The frozen protocol, input manifest, priors, preflight reports, histories, checkpoint hashes and final `REPORT.md` record the experiment. Previous models and datasets are preserved.
