"""Export the completed matched comparison as a standalone scientific figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path(__file__).parent / "pointpillars_head_transfer"
    data = json.loads((out / "comparison.json").read_text())
    results = data["results"]
    keys = [
        "pretrained",
        data["selection"]["real_recipe"] + "_best",
        "synthetic_best",
        "synthetic_coverage_best",
    ]
    labels = [
        "Pretrained\ncontrol",
        "Real\nfine-tune",
        "Synthetic\nfine-tune",
        "Synthetic +\ncoverage\nfine-tune",
    ]
    values = [100 * results[key]["metrics"]["3D_AP50"]["AP101"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    bars = axes[0].bar(
        labels, values, color=["#64748b", "#2563eb", "#ea580c", "#0d9488"]
    )
    axes[0].bar_label(bars, fmt="%.2f%%", padding=4)
    axes[0].set_ylim(0, max(6, max(values) * 1.3))
    axes[0].set_ylabel("3D AP50 (%)")
    axes[0].set_title("Same real test set; validation-selected checkpoints")
    for name, label in [
        ("real_lr1e5", "Head LR 0.00001"),
        ("real_lr1e4", "Head LR 0.0001"),
    ]:
        history = json.loads((out / name / "history.json").read_text())
        rows = [row for row in history if "validation" in row]
        axes[1].plot(
            [row["epoch"] for row in rows],
            [100 * row["validation"]["3D_AP50"]["AP101"] for row in rows],
            marker="o",
            label=label,
        )
    axes[1].set_title("Real-validation learning-rate comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation 3D AP50 (%)")
    axes[1].set_ylim(-0.05, max(1.0, axes[1].get_ylim()[1]))
    axes[1].legend()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Frozen-backbone PointPillars • road_vehicle only", fontsize=14)
    fig.supxlabel(
        "Known diagnostic fold: 27 real test frames / 86 boxes; one seed. Coverage uses real TRAIN priors.",
        fontsize=9,
    )
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    print("HEAD_FIGURE_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
