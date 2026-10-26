"""Check image/label integrity, split leakage, matched budgets and box normalization."""

import json, hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

BASE = Path(__file__).parent
rows = json.loads((BASE / "samples.json").read_text())
summary = json.loads((BASE / "dataset_summary.json").read_text())
groups = defaultdict(set)
hashes = defaultdict(set)
checked = 0
for row in rows:
    image = Path(row["path"])
    label = BASE / "data" / "labels" / (row["name"] + ".txt")
    with Image.open(image) as im:
        w, h = im.size
        assert min(w, h) > 100
        im.verify()
    lines = label.read_text().splitlines()
    assert len(lines) == row["boxes"]
    for line in lines:
        c, x, y, bw, bh = map(float, line.split())
        assert c == 0 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
        assert (
            x - bw / 2 >= -1e-6
            and x + bw / 2 <= 1 + 1e-6
            and y - bh / 2 >= -1e-6
            and y + bh / 2 <= 1 + 1e-6
        )
    if row["domain"] == "real":
        groups[row["split"]].add(row["group"])
        hashes[row["split"]].add(hashlib.sha256(image.read_bytes()).hexdigest())
    checked += 1
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert groups[a].isdisjoint(groups[b]), "Real location leakage"
    assert hashes[a].isdisjoint(hashes[b]), "Duplicate real images across splits"
sets = summary["sets"]
assert sets["real_train"]["images"] == sets["synthetic_train"]["images"] == 63
assert (
    sets["real_train"]["positive_images"] == sets["synthetic_train"]["positive_images"]
)
assert sets["real_test"]["images"] == 40 and sets["real_test"]["positive_images"] == 20
report = {
    "images_and_labels_checked": checked,
    "normalized_boxes_valid": True,
    "real_locations_disjoint": True,
    "no_identical_real_images_across_splits": True,
    "training_image_counts_matched": True,
    "training_positive_image_counts_matched": True,
    "limits": [
        "Visible-mask synthetic boxes and original real boxes still require human policy review.",
        "One synthetic route and few unique vehicles; matched image counts do not imply matched unique-object counts.",
        "This is a small sampled pilot, not the full published OSDaR23 split.",
    ],
}
(BASE / "data_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
