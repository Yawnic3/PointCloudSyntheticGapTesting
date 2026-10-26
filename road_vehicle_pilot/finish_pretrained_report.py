"""Verify the secondary control and summarize the completed transfer experiment."""

import json
from pathlib import Path

from report_pretrained_pointpillars import main as verify_primary


def main():
    out = Path(__file__).parent / "pretrained_pointpillars"
    # Preserve the secondary section while rerunning raw-input/checkpoint checks.
    report_path = out / "REPORT.md"
    secondary = report_path.read_text().split(
        "\n## Predeclared TRAIN-only object-height alignment", 1
    )[1]
    verify_primary()
    primary = json.loads((out / "comparison.json").read_text())["results"]
    height = json.loads((out / "object_height_comparison.json").read_text())["results"]
    for key in ("real_best", "real_last"):
        assert primary[key]["metrics"] == height[key]["metrics"]
    verification_path = out / "final_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["height_probe_real_controls_identical"] = True
    verification["height_probe_all_seven_cases_complete"] = len(height) == 7
    assert all(verification.values())
    verification_path.write_text(json.dumps(verification, indent=2))
    conclusions = """

## Interpretation and next experiment

The published pretrained detector produces some correct 3D detections, but all
three fine-tunes score below its 5.00% primary 3D AP50. The validation-selected
real, synthetic and coverage-augmented synthetic checkpoints score 1.35%, 0.93%
and 0.97%. Coverage adds only 0.04 percentage points over unchanged synthetic.
The final coverage checkpoint reaches 2.22%, but cannot replace the selected
checkpoint based on test performance; at confidence .25 it produces 16 correct
detections and 577 false positives. This experiment has not solved transfer.

The secondary height rule improves the unfine-tuned baseline to 7.26% but reduces
synthetic checkpoint 3D AP50 to zero. A single TRAIN median height translation
does not address the synthetic transfer problem. All secondary real controls
exactly match primary metrics. Original inputs and all frozen weights remain
unchanged after these evaluations.

The next controlled experiment should preserve the pretrained baseline and
compare head-only adaptation with a smaller learning rate, using validation
for selection and a fixed budget. Separately, inspect TRAIN cuboid conventions,
partial vehicle extents and sensor visibility before regenerating synthetic
scenes. These are hypotheses to test, not established causes. Reserve new,
previously unseen real sequences for confirmation; repeated tuning against
this known test fold cannot establish generalization.
"""
    report_path.write_text(
        report_path.read_text()
        + "\n## Predeclared TRAIN-only object-height alignment"
        + secondary
        + conclusions
    )
    print("POST_HEIGHT_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
