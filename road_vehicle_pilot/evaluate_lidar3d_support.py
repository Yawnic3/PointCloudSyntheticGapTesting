"""Apply the predeclared five-return support check, without changing weights."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from train_lidar3d import OUT, FOLD
from lidar3d_geometry import Cuboid, from_seven, evaluate


def main():
    plan = json.loads((OUT / "support_filter_plan.json").read_text())
    assert (
        plan["declared_before_any_primary_test_inference"]
        and plan["minimum_points"] == 5
    )
    assert (
        hashlib.sha256((OUT / "frozen_protocol.json").read_bytes()).hexdigest()
        == plan["primary_protocol_sha256"]
    )
    assert (OUT / "fixed_epoch_comparison.json").exists()
    clouds = {}
    trees = {}
    truth = {}
    for name in (FOLD / "real_test.txt").read_text().splitlines():
        image = Path(name)
        folder = image.parent.parent
        with np.load(folder / "points" / (image.stem + ".npz")) as archive:
            xyz = archive["xyz"]
        valid = (
            (xyz[:, 0] >= 0)
            & (xyz[:, 0] < 160)
            & (np.abs(xyz[:, 1]) < 40)
            & (xyz[:, 2] >= -3)
            & (xyz[:, 2] <= 8)
        )
        xyz = xyz[valid]
        clouds[image.stem] = xyz
        trees[image.stem] = cKDTree(xyz)
        meta = json.loads((folder / "metadata" / (image.stem + ".json")).read_text())
        truth[image.stem] = [o["box"] for o in meta["objects"] if o["evaluated"]]
    report = {"plan": plan, "results": {}}
    for policy, file in [
        ("selected", "test_predictions.json"),
        ("epoch60", "last_test_predictions.json"),
    ]:
        report["results"][policy] = {}
        for kind in ["pillar_cnn", "pillar_transformer"]:
            report["results"][policy][kind] = {}
            for source in ["real", "synthetic"]:
                predictions = json.loads((OUT / kind / source / file).read_text())
                filtered = {}
                before = after = 0
                for name, items in predictions.items():
                    filtered[name] = []
                    for pred in items:
                        before += 1
                        box = Cuboid(from_seven(pred["box"]))
                        index = trees[name].query_ball_point(
                            box.center, float(np.linalg.norm(box.size) / 2 + 0.06)
                        )
                        points = clouds[name][index]
                        count = int(
                            np.all(
                                np.abs((points - box.center) @ box.rotation)
                                <= box.size / 2 + 0.03,
                                axis=1,
                            ).sum()
                        )
                        if count >= 5:
                            filtered[name].append(dict(pred, points_in_cuboid=count))
                            after += 1
                metrics = {
                    "3D_AP25": evaluate(filtered, truth, 0.25),
                    "3D_AP50": evaluate(filtered, truth, 0.5),
                    "3D_AP70": evaluate(filtered, truth, 0.7),
                    "oriented_BEV_AP50": evaluate(filtered, truth, 0.5, "bev"),
                }
                row = {
                    "predictions_before": before,
                    "predictions_after": after,
                    "metrics": metrics,
                }
                report["results"][policy][kind][source] = row
                (
                    OUT
                    / kind
                    / source
                    / (policy + "_support_filtered_predictions.json")
                ).write_text(json.dumps(filtered, indent=2))
                print(
                    "3D_SUPPORT_FILTER_RESULT",
                    policy,
                    kind,
                    source,
                    json.dumps(row),
                    flush=True,
                )
    (OUT / "support_filter_comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "",
        "## Predeclared secondary check: at least five points in predicted cuboid",
        "",
        plan["policy"],
        "",
        "| Checkpoint policy | Model | Source | 3D AP50 | TP / FP / FN at .25 | Predictions retained |",
        "|---|---|---|---:|---|---:|",
    ]
    for policy, models in report["results"].items():
        for kind, sources in models.items():
            for source, row in sources.items():
                m = row["metrics"]["3D_AP50"]
                lines.append(
                    f"| {policy} | {kind} | {source} | {m['AP101']:.2%} | {m['tp']} / {m['fp']} / {m['fn']} | {row['predictions_after']} / {row['predictions_before']} |"
                )
    lines.extend(
        [
            "",
            "This check changes postprocessing, not the trained model or its confidence scores. It cannot recover detections that the model never proposed. Unfiltered selected-checkpoint results remain primary. All test data were previously exposed in the project; this is exploratory development.",
        ]
    )
    path = OUT / "REPORT.md"
    text = path.read_text().split("\n## Predeclared secondary check")[0]
    path.write_text(text + "\n".join(lines) + "\n")
    print("3D_SUPPORT_CHECK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
