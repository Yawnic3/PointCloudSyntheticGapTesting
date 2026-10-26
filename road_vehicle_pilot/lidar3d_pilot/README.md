# Raw-XYZ 3D road-vehicle model exploration

Only `road_vehicle` is trained. The detector sees raw XYZ points, learns point/pillar features, and predicts center XYZ, length, width, height and yaw. RGB, hand-rendered BEV image channels, sensor IDs and annotation IDs are not input features.

Two small experimental models are compared:

- `pillar_cnn`: learned PointNet-style pillar encoder, convolutional multiscale backbone, center-based 3D box head.
- `pillar_transformer`: the same learned point encoder/head, with two local-window self-attention blocks replacing the deepest convolution blocks. Padded shifted windows allow cross-window communication without connecting opposite physical borders.

These are custom research baselines, **not official PointPillars, Swin, 3DETR or pretrained checkpoint replicas**. The design draws on [PointPillars](https://arxiv.org/abs/1812.05784) and [local-window attention](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html). No pretrained weights are used.

## Why this experiment

The earlier YOLO baseline consumed fixed height/density/spread images and predicted only top-down rectangles. This pilot learns features from the point sets and predicts full 3D box dimensions and position. The architecture comparison tests whether adding attention helps this small-data task; it does not assume that transformers will improve it.

## Checks before larger training

Analytical cuboid overlap tests cover identity, separation, translated cubes, 45-degree rotation and tilted-box symmetry. A known box checks decoding coordinates and yaw. Both architectures must pass a training-only subset fit check before full training. The subset has three vehicle boxes; passing it checks implementation behavior, not generalization.

The first FP16 gradient check overflowed. All comparison runs instead use bfloat16 autocast with float32 parameters/losses and finite-gradient checks. SciPy is installed solely for convex-hull geometry; existing dependencies were not upgraded.

## Controlled pilot

The existing retrospective grouped fold 3 supplies 170 training frames per source: 110 positive and 60 negative. Real training groups contain calibration, pedestrian-bridge and Bergedorf vehicles; validation positives are from the underground approach, and test positives from Klein Flottbek. The same validation/test frames are used for all four runs.

Both sources start from identical random parameter values within each architecture, with seed 23, 60 epochs, batch two, AdamW and the same point/geometric augmentation. Source cuboid rotations are transformed consistently with point augmentation. Input clouds and metadata are hashed before training. Selected checkpoints are frozen after all four training runs and before test inference.

Checkpoint selection evaluates real validation every five epochs: maximize 3D AP50, then 3D AP25, then minimize supervised loss. The test does not choose epochs or thresholds. Fixed-threshold counts use confidence 0.25.

## Metrics and limits

The evaluator computes **exact convex 3D cuboid intersection**, retaining the official ground-truth quaternion, including roll/pitch. Predictions are yaw boxes; training approximates roll/pitch with yaw-only targets. AP uses 101 interpolated recall levels and is reported at 3D IoU 0.25, 0.5 and 0.7, plus oriented BEV AP50.

This is an experimental evaluator, **not official OSDaR23 benchmark AP**, and these values are not directly comparable with earlier axis-aligned YOLO BEV scores. Every source location has prior project exposure; this remains diagnostic development rather than fresh blind testing. One seed, one outer fold and a very small set of physical vehicles limit conclusions. Equal frame counts do not equal equal supported object counts or diversity.

## Reproduction

From the `road_vehicle_pilot` folder using the existing project `.venv`:

1. `python train_lidar3d.py --smoke`
2. `python train_lidar3d.py`

`frozen_protocol.json`, `input_manifest.json`, initial weights, per-epoch histories, selected checkpoints, predictions and the final `REPORT.md` preserve the experiment. Earlier YOLO datasets and results are retained.
