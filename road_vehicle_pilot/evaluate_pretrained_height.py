"""Predeclared TRAIN-derived object-height alignment; primary results retained."""

import argparse
import copy
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from official_pointpillars_windows import OUT, OfficialPointPillars
from train_lidar3d import FOLD, Samples
from train_pretrained_pointpillars import assess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = OUT / "object_height_plan.json"
    if not path.exists():
        assert (
            not (OUT / "checkpoints_frozen_before_test.json").exists()
            and not (OUT / "comparison.json").exists()
        )
        priors = json.loads((OUT / "training_priors.json").read_text())
        bottoms = {}
        for source in ["real", "synthetic"]:
            values = []
            for image in (FOLD / (source + "_train.txt")).read_text().splitlines():
                image = Path(image)
                m = json.loads(
                    (
                        image.parent.parent / "metadata" / (image.stem + ".json")
                    ).read_text()
                )
                values.extend(
                    o["box"][2] - o["box"][9] / 2
                    for o in m["objects"]
                    if o["evaluated"]
                )
            bottoms[source] = float(np.median(values))
        shifts = {
            source: bottoms[source]
            + priors["domains"][source]["z_shift"]
            - bottoms["real"]
            for source in ["real", "synthetic"]
        }
        shifts["zero_shot"] = -1.78 - bottoms["real"]
        plan = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "Secondary object-height alignment. Translate real input Z to align its TRAIN median cuboid bottom to the source-domain TRAIN median after the original normalization; restore predicted Z by the inverse shift. No test/validation fitting. Weights and ground truth unchanged. Zero-shot uses published anchor bottom -1.78m as its reference. Primary trackbed-normalized scores remain separate.",
            "source_train_median_bottom_z": bottoms,
            "real_input_z_shift_by_source": shifts,
            "protocol_sha256": sha(OUT / "frozen_protocol.json"),
            "priors_sha256": sha(OUT / "training_priors.json"),
            "runner_sha256": sha(Path(__file__)),
        }
        path.write_text(json.dumps(plan, indent=2))
    plan = json.loads(path.read_text())
    assert sha(Path(__file__)) == plan["runner_sha256"]
    if args.plan_only:
        print("TRAIN_ONLY_HEIGHT_PROBE_DECLARED", json.dumps(plan), flush=True)
        return
    assert (OUT / "comparison.json").exists()
    priors = json.loads((OUT / "training_priors.json").read_text())
    test = Samples(FOLD / "real_test.txt")
    report = {"plan": plan, "results": {}}
    models = [("zero_shot", None)] + [
        (arm, policy)
        for arm in ["real", "synthetic", "synthetic_coverage"]
        for policy in ["best", "last"]
    ]
    for arm, policy in models:
        source = (
            "zero_shot"
            if arm == "zero_shot"
            else "real" if arm == "real" else "synthetic"
        )
        shifted = copy.deepcopy(priors)
        shifted["domains"]["real"]["z_shift"] = plan["real_input_z_shift_by_source"][
            source
        ]
        model = OfficialPointPillars().cuda().eval()
        if policy:
            ckpt = torch.load(
                OUT / "runs" / arm / (policy + ".pt"),
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_official()
        metrics, predictions = assess(model, test, shifted)
        key = arm if policy is None else arm + "_" + policy
        report["results"][key] = {
            "metrics": metrics,
            "input_z_shift": shifted["domains"]["real"]["z_shift"],
        }
        (OUT / (key + "_object_height_predictions.json")).write_text(
            json.dumps(predictions, indent=2)
        )
        print("PRETRAINED_OBJECT_HEIGHT_RESULT", key, json.dumps(metrics), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (OUT / "object_height_comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "",
        "## Predeclared TRAIN-only object-height alignment",
        "",
        plan["policy"],
        "",
        "| Arm / checkpoint | Input Z shift (m) | 3D AP25 | 3D AP50 | Oriented BEV AP50 | TP / FP / FN at .25 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, row in report["results"].items():
        m = row["metrics"]
        f = m["3D_AP50"]
        lines.append(
            f"| {key} | {row['input_z_shift']:.3f} | {m['3D_AP25']['AP101']:.2%} | {m['3D_AP50']['AP101']:.2%} | {m['oriented_BEV_AP50']['AP101']:.2%} | {f['tp']} / {f['fp']} / {f['fn']} |"
        )
    lines.extend(
        [
            "",
            "This is a data-informed preprocessing variant, not a calibrated physical sensor extrinsic. Synthetic models additionally use real TRAIN box-height statistics for inference normalization; this must not be described as zero-real-data transfer. Models/checkpoints are not selected from these test scores.",
        ]
    )
    report_path = OUT / "REPORT.md"
    text = report_path.read_text().split(
        "\n## Predeclared TRAIN-only object-height alignment"
    )[0]
    report_path.write_text(text + "\n".join(lines) + "\n")
    print("PRETRAINED_HEIGHT_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
