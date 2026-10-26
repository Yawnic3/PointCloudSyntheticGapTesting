"""TRAIN-only diagnostic: distinguish fitting failure from location transfer."""

import gc
import json

import numpy as np
import torch

from train_lidar3d import FOLD, Samples
from train_pointpillars_head import OUT, PREVIOUS, new_model
from train_pretrained_pointpillars import assess, sha


def main():
    torch.set_num_threads(4)
    comparison = json.loads((OUT / "comparison.json").read_text())
    data = Samples(FOLD / "real_train.txt")
    tracks = [
        {o["instance_id"] for o in s["objects"] if o["evaluated"]} for s in data.samples
    ]
    uncovered = set().union(*tracks)
    selected = set()
    # Cover every supported training annotation ID before adding spread frames.
    while uncovered:
        index = max(range(len(tracks)), key=lambda i: len(tracks[i] & uncovered))
        assert tracks[index] & uncovered
        selected.add(index)
        uncovered -= tracks[index]
    groups = {}
    negative = []
    for i, sample in enumerate(data.samples):
        if tracks[i]:
            groups.setdefault(sample["id"].split("_")[0], []).append(i)
        else:
            negative.append(i)
    for indices in list(groups.values()) + [negative]:
        count = 6 if indices is negative else 4
        if indices:
            selected.update(
                indices[i]
                for i in np.linspace(
                    0, len(indices) - 1, min(count, len(indices))
                ).astype(int)
            )
    data.samples = [data.samples[i] for i in sorted(selected)]
    plan = {
        "scope": "TRAIN-only fit diagnostic, never held-out performance and never used to select models",
        "selection": "Greedy coverage of all supported training annotation IDs, four spread positive frames per source location and six spread negatives. No prediction-based frame selection.",
        "samples": [sample["id"] for sample in data.samples],
        "physical_annotation_ids": len(set().union(*tracks)),
        "ground_truth_boxes": sum(len(s["truth"]) for s in data.samples),
    }
    (OUT / "training_fit_diagnostic_plan.json").write_text(json.dumps(plan, indent=2))
    priors = json.loads((PREVIOUS / "training_priors.json").read_text())
    winner = comparison["selection"]["real_recipe"]
    paths = {
        "pretrained": None,
        "previous_full_real_best": PREVIOUS / "runs/real/best.pt",
        "previous_full_real_last": PREVIOUS / "runs/real/last.pt",
        "selected_head_real_best": OUT / winner / "best.pt",
        "selected_head_real_last": OUT / winner / "last.pt",
        "higher_rate_head_real_last": OUT / "real_lr1e4/last.pt",
    }
    report = {"plan": plan, "results": {}}
    for name, path in paths.items():
        model = new_model()
        before = sha(path) if path else None
        if path:
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
        metrics, predictions = assess(model, data, priors)
        if path:
            assert sha(path) == before
        report["results"][name] = {"metrics": metrics, "checkpoint_sha256": before}
        (OUT / (name + "_TRAIN_diagnostic_predictions.json")).write_text(
            json.dumps(predictions)
        )
        print("TRAIN_FIT_DIAGNOSTIC", name, json.dumps(metrics["3D_AP50"]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (OUT / "training_fit_diagnostic.json").write_text(json.dumps(report, indent=2))
    lines = [
        "",
        "## TRAIN-only fit diagnostic",
        "",
        plan["scope"],
        "",
        f"{len(data.samples)} training frames, {plan['ground_truth_boxes']} boxes and all {plan['physical_annotation_ids']} supported training annotation IDs. {plan['selection']}",
        "",
        "| Model | TRAIN 3D AP50 | TP / FP / FN at .25 |",
        "|---|---:|---|",
    ]
    for name, row in report["results"].items():
        m = row["metrics"]["3D_AP50"]
        lines.append(
            f"| {name} | {m['AP101']:.2%} | {m['tp']} / {m['fp']} / {m['fn']} |"
        )
    path = OUT / "REPORT.md"
    path.write_text(
        path.read_text().split("\n## TRAIN-only fit diagnostic")[0]
        + "\n".join(lines)
        + "\n"
    )
    print("TRAIN_FIT_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
