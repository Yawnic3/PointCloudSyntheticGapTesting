"""Show the saved validation proposal's horizontal and vertical mismatch."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lidar3d_geometry import Cuboid, EDGES, bev_iou, from_seven, iou3d
from train_lidar3d import FOLD


def main():
    out = Path(__file__).parent / "pointpillars_head_transfer"
    sample = "7_approach_underground_station_7.3__70"
    image = next(
        Path(p)
        for p in (FOLD / "real_val.txt").read_text().splitlines()
        if Path(p).stem == sample
    )
    root = image.parent.parent
    meta = json.loads((root / "metadata" / (sample + ".json")).read_text())
    predictions = json.loads((out / "pretrained_validation.json").read_text())[
        "predictions"
    ][sample]
    truth = [Cuboid(o["box"]) for o in meta["objects"] if o["evaluated"]]
    pairs = [
        (bev_iou(g, Cuboid(from_seven(p["box"]))), g, p)
        for g in truth
        for p in predictions
    ]
    _, gt, proposal = max(pairs, key=lambda item: item[0])
    pred = Cuboid(from_seven(proposal["box"]))
    with np.load(root / "points" / (sample + ".npz")) as data:
        xyz = data["xyz"]
    crop = xyz[
        (np.abs(xyz[:, 0] - gt.center[0]) < 7)
        & (np.abs(xyz[:, 1] - gt.center[1]) < 5)
        & (xyz[:, 2] > -2)
        & (xyz[:, 2] < 5)
    ]
    if len(crop) > 25000:
        crop = crop[np.random.default_rng(23).choice(len(crop), 25000, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for axis, dims, title in [
        (axes[0], (0, 1), "Top view: X / Y"),
        (axes[1], (0, 2), "Side view: X / Z"),
    ]:
        a, b = dims
        axis.scatter(
            crop[:, a], crop[:, b], s=2, c="#94a3b8", alpha=0.5, rasterized=True
        )
        for box, color, label in [
            (gt, "#15803d", "Ground truth"),
            (pred, "#ea580c", "Pretrained proposal"),
        ]:
            for index, (start, end) in enumerate(EDGES):
                values = box.corners[[start, end]]
                axis.plot(
                    values[:, a],
                    values[:, b],
                    color=color,
                    linewidth=1.8,
                    label=label if index == 0 else None,
                )
        axis.set_xlabel("Forward X (m)")
        axis.set_ylabel("Left Y (m)" if b == 1 else "Height Z (m)")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper right")
    fig.suptitle(
        f"Validation example • BEV IoU {bev_iou(gt, pred):.2f}, 3D IoU {iou3d(gt, pred):.2f}\nPredicted center is {gt.center[2] - pred.center[2]:.2f} m below the annotation",
        fontsize=13,
    )
    fig.supxlabel(
        "Saved pretrained-control output; illustrative validation case, not a performance estimate.",
        fontsize=9,
    )
    fig.savefig(out / "validation_height_failure.png", dpi=180)
    plt.close(fig)
    print("VALIDATION_HEIGHT_FIGURE_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
