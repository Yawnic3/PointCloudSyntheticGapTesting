"""Diagnose existing pilot models without changing the held-out benchmark."""

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
OUT = BASE / "diagnostics"
OUT.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO
from run_pilot import iou


def truth(path):
    with Image.open(path) as im:
        width, height = im.size
    boxes = []
    for line in (
        (BASE / "data" / "labels" / (Path(path).stem + ".txt")).read_text().splitlines()
    ):
        _, x, y, w, h = map(float, line.split())
        boxes.append(
            (
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            )
        )
    return boxes, width, height


def crop_review(paths, name):
    choices = []
    for path in paths:
        boxes, width, height = truth(path)
        for box in boxes:
            choices.append(((box[2] - box[0]) * (box[3] - box[1]), path, box))
    choices.sort(reverse=True)
    sheet = Image.new("RGB", (1200, 600), "#222222")
    draw = ImageDraw.Draw(sheet)
    for index, (_, path, box) in enumerate(choices[:8]):
        image = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = box
        padding = max(25, (x1 - x0) * 0.5, (y1 - y0) * 0.5)
        bounds = (
            max(0, int(x0 - padding)),
            max(0, int(y0 - padding)),
            min(image.width, int(x1 + padding)),
            min(image.height, int(y1 + padding)),
        )
        image = image.crop(bounds)
        d = ImageDraw.Draw(image)
        d.rectangle(
            (x0 - bounds[0], y0 - bounds[1], x1 - bounds[0], y1 - bounds[1]),
            outline="lime",
            width=2,
        )
        image.thumbnail((300, 255))
        x, y = index % 4 * 300, index // 4 * 300
        sheet.paste(image, (x, y + 40))
        draw.text((x + 4, y + 4), f"GT {x1-x0:.0f} x {y1-y0:.0f} px", fill="white")
        draw.text((x + 4, y + 20), Path(path).stem[:35], fill="white")
    sheet.save(OUT / (name + "_label_crops.jpg"))


def predictions(model, paths, name, coco=False):
    thresholds = [0.001, 0.01, 0.05, 0.25]
    totals = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    per_object = []
    kwargs = {"classes": [2, 5, 7]} if coco else {}
    results = model.predict(
        paths,
        imgsz=960,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=0,
        stream=True,
        verbose=False,
        **kwargs,
    )
    positive_saved = 0
    for result in results:
        gt, width, height = truth(result.path)
        rows = sorted(result.boxes.data.cpu().tolist(), key=lambda r: -r[4])
        for threshold in thresholds:
            matches = set()
            for row in rows:
                if row[4] < threshold:
                    continue
                candidates = [
                    (iou(row[:4], box), j)
                    for j, box in enumerate(gt)
                    if j not in matches
                ]
                if candidates and max(candidates)[0] >= 0.5:
                    matches.add(max(candidates)[1])
                    totals[str(threshold)]["tp"] += 1
                else:
                    totals[str(threshold)]["fp"] += 1
            totals[str(threshold)]["fn"] += len(gt) - len(matches)
        for box in gt:
            overlaps = [(iou(r[:4], box), r[4]) for r in rows]
            best = max(overlaps, default=(0, 0))
            per_object.append(
                {
                    "image": Path(result.path).name,
                    "best_iou": best[0],
                    "confidence_at_best_iou": best[1],
                    "input_width": (box[2] - box[0]) * 960 / max(width, height),
                    "input_height": (box[3] - box[1]) * 960 / max(width, height),
                }
            )
        if gt and positive_saved < 4:
            image = Image.open(result.path).convert("RGB")
            d = ImageDraw.Draw(image)
            for box in gt:
                d.rectangle(box, outline="lime", width=3)
            for row in rows[:20]:
                if row[4] >= 0.01:
                    d.rectangle(row[:4], outline="red", width=2)
                    d.text((row[0], row[1]), f"{row[4]:.3f}", fill="red")
            image.thumbnail((1400, 1000))
            image.save(OUT / f"{name}_positive_{positive_saved}.jpg")
            positive_saved += 1
    for counts in totals.values():
        counts["precision"] = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        counts["recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return {"thresholds": totals, "objects": per_object}


def main():
    lists = {
        k: (BASE / "data" / (k + ".txt")).read_text().splitlines()
        for k in ["real_train", "synthetic_train", "real_val", "real_test"]
    }
    report = {
        "purpose": "Diagnostic exploration, not new benchmark tuning",
        "box_sizes": {},
        "models": {},
    }
    for split, paths in lists.items():
        dimensions = []
        for path in paths:
            boxes, w, h = truth(path)
            dimensions.extend(
                [(b[2] - b[0]) * 960 / max(w, h), (b[3] - b[1]) * 960 / max(w, h)]
                for b in boxes
            )
        arr = np.asarray(dimensions)
        report["box_sizes"][split] = {
            "objects": len(arr),
            "median_input_wh": np.median(arr, axis=0).tolist(),
            "fraction_min_side_under_8px": float(np.mean(arr.min(axis=1) < 8)),
        }
        crop_review(paths, split)
    for domain in ["real", "synthetic"]:
        model = YOLO(str(BASE / "runs" / ("pilot30_" + domain) / "weights" / "last.pt"))
        for split in [domain + "_train", "real_val", "real_test"]:
            name = domain + "_on_" + split
            report["models"][name] = predictions(model, lists[split], name)
            (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2))
            print(name, json.dumps(report["models"][name]["thresholds"]), flush=True)
        csv_path = BASE / "runs" / ("pilot30_" + domain) / "results.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        report.setdefault("curves", {})[domain] = {
            "first": rows[0],
            "last": rows[-1],
            "max_validation_AP50": max(float(r["metrics/mAP50(B)"]) for r in rows),
        }
    pretrained = YOLO(str(BASE / "models" / "yolo26s.pt"))
    report["models"]["coco_initial_on_real_val"] = predictions(
        pretrained, lists["real_val"], "coco_on_real_val", coco=True
    )
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print("DIAGNOSIS COMPLETE", json.dumps(report["box_sizes"]), flush=True)


if __name__ == "__main__":
    main()
