"""Secondary fixed-epoch evaluation, declared before primary test inference."""

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from train_lidar3d import OUT, FOLD, Samples, assess
from lidar3d_models import PillarDetector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen = OUT / "fixed_epoch_secondary_plan.json"
    if not frozen.exists():
        assert not (OUT / "selected_checkpoints_before_test.json").exists()
        assert not (OUT / "comparison.json").exists()
        frozen.write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "policy": "Secondary evaluation of last.pt at fixed epoch 60 for all four runs; primary best-validation comparison unchanged. Declared before primary test inference.",
                    "primary_protocol_sha256": sha(OUT / "frozen_protocol.json"),
                    "runner_sha256": sha(Path(__file__)),
                    "checkpoint_epoch": 60,
                    "models": ["pillar_cnn", "pillar_transformer"],
                    "sources": ["real", "synthetic"],
                },
                indent=2,
            )
        )
    plan = json.loads(frozen.read_text())
    assert plan["runner_sha256"] == sha(Path(__file__)) and plan[
        "primary_protocol_sha256"
    ] == sha(OUT / "frozen_protocol.json")
    if args.plan_only:
        print("FIXED_EPOCH_SECONDARY_DECLARED", frozen, flush=True)
        return
    assert (OUT / "comparison.json").exists()
    data = Samples(FOLD / "real_test.txt")
    report = {"plan": plan, "results": {}}
    for kind in plan["models"]:
        report["results"][kind] = {}
        for source in plan["sources"]:
            folder = OUT / kind / source
            path = folder / "last.pt"
            before = sha(path)
            complete = json.loads((folder / "completed.json").read_text())
            assert complete["epochs"] == 60 and complete["last_sha256"] == before
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == 60
            model = PillarDetector(kind).cuda()
            model.load_state_dict(checkpoint["state_dict"])
            metrics, predictions = assess(model, data, True)
            assert sha(path) == before
            report["results"][kind][source] = {
                "checkpoint_sha256": before,
                "metrics": metrics,
            }
            (folder / "last_test_predictions.json").write_text(
                json.dumps(predictions, indent=2)
            )
            print(
                "3D_FIXED_EPOCH_RESULT", kind, source, json.dumps(metrics), flush=True
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    (OUT / "fixed_epoch_comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "",
        "## Predeclared secondary comparison: fixed epoch 60",
        "",
        plan["policy"],
        "",
        "| Model | Training source | 3D AP25 | 3D AP50 | 3D AP70 | Oriented BEV AP50 | TP / FP / FN at .25, 3D IoU .5 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for kind, sources in report["results"].items():
        for source, row in sources.items():
            m = row["metrics"]
            f = m["3D_AP50"]
            lines.append(
                f"| {kind} | {source} | {m['3D_AP25']['AP101']:.2%} | {m['3D_AP50']['AP101']:.2%} | {m['3D_AP70']['AP101']:.2%} | {m['oriented_BEV_AP50']['AP101']:.2%} | {f['tp']} / {f['fp']} / {f['fn']} |"
            )
    path = OUT / "REPORT.md"
    text = path.read_text().split("\n## Predeclared secondary comparison")[0]
    path.write_text(text + "\n".join(lines) + "\n")
    print("3D_FIXED_EPOCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
