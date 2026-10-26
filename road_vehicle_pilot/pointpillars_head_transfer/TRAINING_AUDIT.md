# Training controls and input audit

The experiment changes the trainable parameters and learning rate while keeping
the previous data, geometry, loss, augmentation and inference settings fixed.
Only the 7,700 detection-head parameters receive gradients. All feature weights
and batch-normalization buffers are checked against the published initialization.

The deterministic TRAIN-only retention replay contains 175 real and 557
synthetic supported boxes. No real box falls below five points after the
160,000-point limit and 32,000-voxel/32-point-per-voxel limits. One synthetic box
does. Real yaw-only projections have a minimum exact 3D overlap of 95.10% with
the source cuboids and a median of 100%. These checks rule out severe losses
from those particular operations in this replay; they do not validate every
stochastic transformation or every aspect of the runtime adapter.

Median retained in-cuboid point fractions are 98.58% for real and 87.50% for
synthetic boxes. After these limits, 14.86% of real and 11.13% of unchanged
synthetic training boxes have fewer than 50 points. Counts include all points
inside annotated cuboids, which can include ground or clutter.

The two sources have equal frame counts (170), positive frames (110), negative
frames (60), initialization and update budgets. They do not have equal box or
physical-object counts. Real training has 15 supported annotation IDs across
three positive locations. Validation has four IDs from one location, making
checkpoint selection sensitive to that location. No frames or labels move
between splits in this experiment.

The archived official layers, anchors and coder were checked in the preceding
experiment. This Windows runtime still uses custom CPU hard voxelization/NMS
and adapted railway support; it is not a run inside the full MMDetection3D
framework. Published weights use KITTI Car pretraining, and the new runs are
fine-tunes of that common starting point.

One further training-policy difference deserves a separate controlled test:
[MMDetection3D's dataset defaults to filtering empty-GT training samples](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/datasets/det3d_dataset.py),
whereas this pilot retains 60 negative training frames. The
[reference KITTI pipeline](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/_base_/datasets/kitti-3d-3class.py)
also includes database object sampling and object noise, which this pilot does
not reproduce. These differences are hypotheses, not proof of the cause of
poor detection. They are left unchanged here to isolate head-only adaptation.

See `training_input_audit.json` for per-frame and per-box retention records.
