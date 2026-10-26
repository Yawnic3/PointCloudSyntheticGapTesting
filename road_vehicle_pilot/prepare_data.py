"""Prepare a small road_vehicle-only RGB pilot with location-disjoint real splits."""

import json, random, shutil, csv
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from fetch_real import archive

BASE = Path(__file__).parent
DATA = BASE / "data"
SYN = BASE.parent / "moving_dataset" / "captures" / "rgb"


def write_item(name, source, boxes):
    image = DATA / "images" / f"{name}.png"
    label = DATA / "labels" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(exist_ok=True)
    if source.resolve() != image.resolve():
        shutil.copy2(source, image)
    label.write_text(
        "\n".join("0 " + " ".join(f"{v:.8f}" for v in b) for b in boxes)
        + ("\n" if boxes else "")
    )
    return str(image.resolve())


def synthetic():
    palette = json.loads((BASE / "synthetic_masks" / "palette.json").read_text())
    colors = np.asarray(list(palette.values()), dtype=np.int16)
    ids = list(palette)
    records = []
    for source in sorted(SYN.glob("*.png")):
        mask = BASE / "synthetic_masks" / source.name
        assert mask.exists(), f"Mask missing: {mask}"
        pixels = np.asarray(Image.open(mask).convert("RGB")).astype(np.int16)
        h, w = pixels.shape[:2]
        candidate = (pixels[:, :, 0] >= 250) & (np.abs(pixels[:, :, 2] - 128) <= 4)
        boxes = []
        visible = []
        # Classify exact emission colors; reject partially covered edge samples.
        for i, color in zip(ids, colors):
            selected = candidate & (np.max(np.abs(pixels - color), axis=2) <= 3)
            yy, xx = np.where(selected)
            if len(xx) < 3:
                continue
            x0, x1 = xx.min(), xx.max() + 1
            y0, y1 = yy.min(), yy.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append(
                ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
            )
            visible.append(i)
        name = "synthetic_" + source.stem
        path = write_item(name, source, boxes)
        records.append(
            {
                "name": name,
                "path": path,
                "domain": "synthetic",
                "split": "train",
                "sequence": "single_synthetic_route",
                "positive": bool(boxes),
                "boxes": len(boxes),
                "visible_instances": visible,
            }
        )
    return records


def sample_balanced(pool, npos, nneg, seed):
    rng = random.Random(seed)

    # Round-robin sequences reduces domination by long sequences.
    def pick(positive, n):
        grouped = defaultdict(list)
        for row in pool:
            if bool(row["frame"]["boxes"]) == positive:
                grouped[row["sequence"]].append(row)
        for rows in grouped.values():
            rng.shuffle(rows)
        keys = list(grouped)
        rng.shuffle(keys)
        out = []
        while len(out) < n:
            progress = False
            for key in keys:
                if grouped[key] and len(out) < n:
                    out.append(grouped[key].pop())
                    progress = True
            if not progress:
                raise ValueError(
                    f"Insufficient frames: need {n} positive={positive}, got {len(out)}"
                )
        return out

    result = pick(True, npos) + pick(False, nneg)
    rng.shuffle(result)
    return result


def fetch_sequence(args):
    name, rows = args
    folder = BASE / "real_raw" / name / "rgb_center"
    folder.mkdir(exist_ok=True)
    with archive(name) as z:
        members = set(z.namelist())
        for row in rows:
            member = row["frame"]["image_member"]
            dest = folder / Path(member).name
            if not dest.exists():
                if member not in members:
                    raise ValueError(f"Missing {member}")
                dest.write_bytes(z.read(member))
        # Preserve official readme/license, with the selected public-data samples.
        for member in ["license.md", "readme.md"]:
            if member in members:
                (BASE / "real_raw" / name / member).write_bytes(z.read(member))
    print("DOWNLOADED", name, len(rows), flush=True)


def real(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_sequence, groups.items()))
    result = []
    for row in rows:
        frame = row["frame"]
        source = (
            BASE
            / "real_raw"
            / row["sequence"]
            / "rgb_center"
            / Path(frame["image_member"]).name
        )
        with Image.open(source) as im:
            w, h = im.size
        boxes = []
        for b in frame["boxes"]:
            x, y, bw, bh = b["bbox_cxcywh_pixels"]
            x0 = max(0, x - bw / 2)
            x1 = min(w, x + bw / 2)
            y0 = max(0, y - bh / 2)
            y1 = min(h, y + bh / 2)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Invalid real bbox")
            boxes.append(
                ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
            )
        name = "real_" + row["sequence"] + "_" + frame["frame_id"]
        path = write_item(name, source, boxes)
        result.append(
            {
                "name": name,
                "path": path,
                "domain": "real",
                "split": row["split"],
                "sequence": row["sequence"],
                "group": row["group"],
                "positive": bool(boxes),
                "boxes": len(boxes),
                "source_instance_ids": [b["instance_id"] for b in frame["boxes"]],
            }
        )
    return result


def review(records):
    chosen = []
    for domain in ["real", "synthetic"]:
        chosen.extend(
            [r for r in records if r["domain"] == domain and r["positive"]][:6]
        )
    sheet = Image.new("RGB", (1280, 3 * 250), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(chosen):
        im = Image.open(r["path"]).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for line in (DATA / "labels" / (r["name"] + ".txt")).read_text().splitlines():
            _, x, y, bw, bh = map(float, line.split())
            d.rectangle(
                (
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                ),
                outline="red",
                width=3,
            )
        im.thumbnail((320, 220))
        x = (i % 4) * 320
        y = (i // 4) * 250
        sheet.paste(im, (x, y + 30))
        draw.text(
            (x + 4, y + 4),
            r["domain"] + " " + r["split"] + " " + str(r["boxes"]) + " boxes",
            fill="white",
        )
    sheet.save(BASE / "label_review.jpg")


if __name__ == "__main__":
    syn = synthetic()
    npos = sum(r["positive"] for r in syn)
    nneg = len(syn) - npos
    inventory = json.loads((BASE / "real_inventory.json").read_text())["sequences"]
    # Entire location groups are held apart, including all subsequences from each location.
    split_groups = {
        "train": {1, 3, 4, 5, 7, 12, 15, 20},
        "val": {6, 10, 14, 18},
        "test": {2, 8, 9, 11, 13, 16, 17, 19, 21},
    }
    pools = defaultdict(list)
    for seq in inventory:
        split = next(k for k, v in split_groups.items() if seq["group"] in v)
        pools[split].extend(
            {
                "sequence": seq["sequence"],
                "group": seq["group"],
                "frame": f,
                "split": split,
            }
            for f in seq["frames"]
        )
    selected = (
        sample_balanced(pools["train"], npos, nneg, 23)
        + sample_balanced(pools["val"], 10, 10, 24)
        + sample_balanced(pools["test"], 20, 20, 25)
    )
    assert not (
        split_groups["train"] & split_groups["val"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["val"] & split_groups["test"]
    )
    (BASE / "selected_real_frames.json").write_text(json.dumps(selected, indent=2))
    records = syn + real(selected)
    sets = {
        "synthetic_train": [r for r in records if r["domain"] == "synthetic"],
        "real_train": [
            r for r in records if r["domain"] == "real" and r["split"] == "train"
        ],
        "real_val": [
            r for r in records if r["domain"] == "real" and r["split"] == "val"
        ],
        "real_test": [
            r for r in records if r["domain"] == "real" and r["split"] == "test"
        ],
    }
    for name, rows in sets.items():
        (DATA / (name + ".txt")).write_text("\n".join(r["path"] for r in rows) + "\n")
    for name in ["real", "synthetic"]:
        (BASE / (name + ".yaml")).write_text(
            f"path: {DATA.resolve().as_posix()}\ntrain: {name}_train.txt\nval: real_val.txt\ntest: real_test.txt\nnames:\n  0: road_vehicle\n"
        )
    (BASE / "samples.json").write_text(json.dumps(records, indent=2))
    summary = {
        name: {
            "images": len(rows),
            "positive_images": sum(r["positive"] for r in rows),
            "boxes": sum(r["boxes"] for r in rows),
            "sequences": sorted(set(r["sequence"] for r in rows)),
        }
        for name, rows in sets.items()
    }
    (BASE / "dataset_summary.json").write_text(
        json.dumps(
            {
                "sets": summary,
                "matched_training_image_count": True,
                "matched_training_positive_image_count": True,
                "split_groups": {k: sorted(v) for k, v in split_groups.items()},
                "synthetic_policy": "Visible mask boxes; >=3 visible pixels and >=2 pixels each dimension. Fully occluded objects excluded. Real labels retained from official annotations, clipped to image. Requires visual review; conventions not claimed equivalent.",
                "purpose": "Pipeline pilot, not a final sim-to-real benchmark",
            },
            indent=2,
        )
    )
    review(records)
    print("DATA READY", json.dumps(summary), flush=True)
