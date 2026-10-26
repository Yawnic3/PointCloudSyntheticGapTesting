"""Render a TRAIN-only augmentation replay for visual inspection."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_pointpillars_object_cv import OUT
from lidar3d_geometry import Cuboid, EDGES, from_seven


def main():
    with np.load(OUT / "TRAIN_augmentation_example.npz") as data:
        before, after = data["before"], data["after"]
        old, boxes = data["original_boxes"], data["boxes"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for col, (points, labels, title) in enumerate(
        [
            (before, old, "Before object augmentation"),
            (after, boxes, "After copy/paste and height augmentation"),
        ]
    ):
        if len(points) > 35000:
            points = points[
                np.random.default_rng(23).choice(len(points), 35000, replace=False)
            ]
        for row, dim in enumerate([1, 2]):
            axis = axes[row, col]
            axis.scatter(
                points[:, 0],
                points[:, dim],
                s=0.5,
                color="#94a3b8",
                alpha=0.5,
                rasterized=True,
            )
            for b in labels:
                is_original = any(
                    np.allclose(np.asarray(b)[:2], np.asarray(other)[:2])
                    for other in old
                )
                color = "#2563eb" if col == 0 or is_original else "#ea580c"
                vertices = Cuboid(from_seven(b)).corners
                for a, c in EDGES:
                    values = vertices[[a, c]]
                    axis.plot(values[:, 0], values[:, dim], color=color, linewidth=1.3)
            axis.set_xlim(0, 160)
            axis.set_ylim((-40, 40) if dim == 1 else (-3, 8))
            axis.set_xlabel("Forward X (m)")
            axis.set_ylabel("Left Y (m)" if dim == 1 else "Height Z (m)")
            axis.grid(alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(title)
    fig.suptitle(
        "TRAIN-only replay • blue: existing boxes • orange: pasted boxes",
        fontsize=14,
    )
    fig.supxlabel(
        "Source TRAIN returns and labels only. Point-level augmentation; no physical reraycast or matching RGB.",
        fontsize=10,
    )
    fig.savefig(OUT / "TRAIN_augmentation_example.png", dpi=180)
    plt.close(fig)
    print("OBJECT_AUG_FIGURE_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
