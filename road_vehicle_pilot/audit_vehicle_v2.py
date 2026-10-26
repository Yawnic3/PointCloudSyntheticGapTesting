"""Independently audit RGBs, index masks, YOLO boxes and route separation."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
ROOT = BASE / "synthetic_v2"


def contact_sheet(records, destination):
    selected = []
    for family in [
        "hatchback",
        "sedan",
        "estate",
        "suv",
        "pickup",
        "van",
        "box_truck",
        "bus",
    ]:
        choices = [
            (o["visible_pixels"], r, o)
            for r in records
            for o in r["objects"]
            if o["family"] == family and o["bbox_xyxy_pixels"]
        ]
        choices.sort(key=lambda item: -item[0])
        if choices:
            selected.append((family, choices[0][1], choices[0][2]))
    sheet = Image.new("RGB", (1600, 650), "#222222")
    draw = ImageDraw.Draw(sheet)
    for index, (family, record, obj) in enumerate(selected):
        im = Image.open(ROOT / "images" / (record["sample_id"] + ".png")).convert("RGB")
        box = obj["bbox_xyxy_pixels"]
        pad = max(box[2] - box[0], box[3] - box[1]) * 0.4
        left, top = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
        right, bottom = min(im.width, int(box[2] + pad)), min(
            im.height, int(box[3] + pad)
        )
        im = im.crop((left, top, right, bottom))
        ImageDraw.Draw(im).rectangle(
            (box[0] - left, box[1] - top, box[2] - left, box[3] - top),
            outline="lime",
            width=2,
        )
        im.thumbnail((400, 280))
        x, y = index % 4 * 400, index // 4 * 325
        sheet.paste(im, (x, y + 40))
        draw.text((x + 5, y + 5), family + " | visible label", fill="white")
        draw.text((x + 5, y + 22), record["sample_id"], fill="white")
    sheet.save(destination)


def scene_sheet(records, destination):
    rows = []
    for route in range(12):
        options = [r for r in records if r["route_id"] == route and r["objects"]]
        if options:
            rows.append(options[min(2, len(options) - 1)])
    sheet = Image.new("RGB", (1600, 3 * 255), "#222222")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(rows):
        im = Image.open(ROOT / "images" / (record["sample_id"] + ".png")).convert("RGB")
        d = ImageDraw.Draw(im)
        for obj in record["objects"]:
            if obj["bbox_xyxy_pixels"]:
                d.rectangle(obj["bbox_xyxy_pixels"], outline="lime", width=2)
        im.thumbnail((400, 225))
        x, y = index % 4 * 400, index // 4 * 255
        sheet.paste(im, (x, y + 30))
        draw.text(
            (x + 5, y + 5),
            f'{record["sample_id"]} {record["split"]} {record["context"]}',
            fill="white",
        )
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    records = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "metadata").glob("route_*.json"))
    ]
    if not args.partial:
        expected = {
            f"route_{route:02}_{frame:04}" for route in range(12) for frame in range(40)
        }
        assert {
            r["sample_id"] for r in records
        } == expected, "Missing or unexpected captures"
    summaries = {}
    identities = set()
    source_hashes = defaultdict(set)
    geometries = set()
    verified_masks = 0
    for split in ["train", "val", "test"]:
        subset = [r for r in records if r["split"] == split]
        sizes = []
        families = Counter()
        bands = Counter()
        family_sizes = defaultdict(list)
        actual_ranges = Counter()
        boxes = 0
        empty = 0
        visible_ids = set()
        requested_objects = 0
        contexts = set()
        for record in subset:
            sample = record["sample_id"]
            image_path = ROOT / "images" / (sample + ".png")
            with Image.open(image_path) as im:
                assert im.size == (1280, 720)
                im.verify()
            source_hashes[split].add(
                hashlib.sha256(image_path.read_bytes()).hexdigest()
            )
            contexts.add(record["context"])
            text = (ROOT / "labels" / (sample + ".txt")).read_text().splitlines()
            wanted = [o for o in record["objects"] if o["bbox_xyxy_pixels"]]
            assert len(text) == len(wanted)
            if record["negative_requested"]:
                assert not text and not record["objects"]
            empty += not text
            requested_objects += len(record["objects"])
            if not args.partial:
                frame = int(sample.rsplit("_", 1)[1]) + 1
                exr = ROOT / "masks" / f"{sample}_{frame:04}.exr"
                file = OpenEXR.File(str(exr))
                mask = file.channels()["V"].pixels
                assert mask.shape == (720, 1280) and np.isfinite(mask).all()
                assert np.allclose(mask, np.rint(mask))
                assert set(np.unique(mask).tolist()) <= {
                    0,
                    *[o["mask_id"] for o in record["objects"]],
                }
                for obj in record["objects"]:
                    yy, xx = np.where(mask == obj["mask_id"])
                    assert len(xx) == obj["visible_pixels"]
                    actual = None
                    if len(xx) >= 3:
                        bounds = [
                            int(xx.min()),
                            int(yy.min()),
                            int(xx.max()) + 1,
                            int(yy.max()) + 1,
                        ]
                        if bounds[2] - bounds[0] >= 2 and bounds[3] - bounds[1] >= 2:
                            actual = bounds
                    assert actual == obj["bbox_xyxy_pixels"], (sample, obj["mask_id"])
                verified_masks += 1
            for obj in record["objects"]:
                assert obj["instance_id"] not in identities, "Reused actor identity"
                identities.add(obj["instance_id"])
                geometries.add(
                    (
                        obj["family"],
                        *[round(v, 3) for v in obj["dimensions_m"]],
                        obj["paint"],
                    )
                )
            for line, obj in zip(text, wanted):
                class_id, x, y, w, h = map(float, line.split())
                assert (
                    class_id == 0
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                )
                assert (
                    x - w / 2 >= -1e-7
                    and x + w / 2 <= 1 + 1e-7
                    and y - h / 2 >= -1e-7
                    and y + h / 2 <= 1 + 1e-7
                )
                a, b, c, d = obj["bbox_xyxy_pixels"]
                expected = np.array(
                    [(a + c) / 2560, (b + d) / 1440, (c - a) / 1280, (d - b) / 720]
                )
                assert np.allclose([x, y, w, h], expected, atol=1e-7)
                sizes.append([w * 960, h * 540])
                family_sizes[obj["family"]].append([w * 960, h * 540])
                distance = obj["range_m"]
                range_bin = (
                    "0-30"
                    if distance < 30
                    else (
                        "30-60"
                        if distance < 60
                        else "60-100" if distance < 100 else "100+"
                    )
                )
                actual_ranges[range_bin] += 1
                families[obj["family"]] += 1
                bands[obj["distance_band"]] += 1
                visible_ids.add(obj["instance_id"])
                boxes += 1
        a = np.asarray(sizes)
        summaries[split] = {
            "images": len(subset),
            "positive_images": len(subset) - empty,
            "negative_images": empty,
            "visible_boxes": boxes,
            "created_vehicle_arrangements": requested_objects,
            "visible_instance_ids": len(visible_ids),
            "routes": sorted({r["route_id"] for r in subset}),
            "contexts": sorted(contexts),
            "families": dict(families),
            "requested_placement_bands": dict(bands),
            "actual_range_bins_m": dict(actual_ranges),
            "family_size_statistics": {
                family: {
                    "median_input_wh_at_960": np.median(
                        np.asarray(values), axis=0
                    ).tolist(),
                    "fraction_min_side_under_8px": float(
                        np.mean(np.asarray(values).min(axis=1) < 8)
                    ),
                }
                for family, values in family_sizes.items()
            },
            "median_input_wh_at_960": np.median(a, axis=0).tolist() if len(a) else [],
            "fraction_min_side_under_8px": (
                float(np.mean(a.min(axis=1) < 8)) if len(a) else None
            ),
            "fraction_min_side_at_least_16px": (
                float(np.mean(a.min(axis=1) >= 16)) if len(a) else None
            ),
        }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(summaries[a]["routes"]).isdisjoint(summaries[b]["routes"])
        assert source_hashes[a].isdisjoint(
            source_hashes[b]
        ), "Duplicate images cross splits"
    report = {
        "complete": not args.partial,
        "splits": summaries,
        "unique_actor_arrangements": len(identities),
        "distinct_geometry_paint_recipes_rounded_1mm": len(geometries),
        "independent_masks_verified": verified_masks,
        "scope": "Parametric vehicle variants and independent traffic arrangements, not manufacturer CAD assets or a tracking dataset.",
        "original_pilot_preserved": True,
        "real_test_used_for_scoring": False,
        "annotation_policy": "Visible index-mask extents; original real annotation policies still need alignment.",
    }
    if not args.partial:
        assert summaries["train"]["images"] == 320
        for route in range(12):
            for frame in [0, 39]:
                snapshot = ROOT / "scenes" / f"route_{route:02}_{frame:04}.blend"
                assert snapshot.is_file() and snapshot.stat().st_size > 100000, snapshot
        assert len(summaries["train"]["families"]) == 8
        assert (
            summaries["train"]["fraction_min_side_under_8px"] < 0.20
        ), "Too many tiny training labels"
        assert (
            summaries["train"]["median_input_wh_at_960"][1] >= 16
        ), "Median target height too small"
        for split in ["train", "val", "test"]:
            subset = [r for r in records if r["split"] == split]
            (ROOT / (split + ".txt")).write_text(
                "\n".join(
                    str((ROOT / "images" / (r["sample_id"] + ".png")).resolve())
                    for r in subset
                )
                + "\n"
            )
        config = f"path: {ROOT.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: road_vehicle\n"
        (ROOT / "data.yaml").write_text(config)
        (BASE / "synthetic_v2.yaml").write_text(config)
        (ROOT / "audit.json").write_text(json.dumps(report, indent=2))
        provenance = {
            "generator_sha256": hashlib.sha256(
                (BASE / "generate_vehicle_v2.py").read_bytes()
            ).hexdigest(),
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "blender_version": "4.2.3 LTS",
            "seed_rule": "route = 80231 + route_id * 1031; arrangement = route_id * 100000 + frame_id + 9900000",
            "images": len(records),
            "manifest_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()
            ).hexdigest(),
            "original_real_test_notice": "Previously inspected during diagnosis; no new model test scores were used or computed for this generation batch.",
        }
        (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2))
        contact_sheet(records, ROOT / "vehicle_review.jpg")
        scene_sheet(records, ROOT / "route_review.jpg")
        training = summaries["train"]
        family_rows = "\n".join(
            f"| {family} | {count} |"
            for family, count in sorted(training["families"].items())
        )
        width, height = training["median_input_wh_at_960"]
        small = training["fraction_min_side_under_8px"] * 100
        overview = f"""# Completed synthetic road-vehicle v2 dataset

480 images: 320 train, 80 validation, 80 reserved synthetic test.
Twelve independently seeded routes; entire routes belong to only one split.
All 480 RGB/EXR pairs, visible boxes and normalized YOLO labels passed the independent audit.

| Training-data property | Original pilot | V2 |
| --- | ---: | ---: |
| Images | 63 | 320 |
| Visible box occurrences | 107 | {training['visible_boxes']} |
| Median box width at input 960 | 12.75 px | {width:.2f} px |
| Median box height at input 960 | 10.5 px | {height:.2f} px |
| Boxes with a side under 8 px | 44.9% | {small:.1f}% |

The v2 training split contains {training['positive_images']} positive and {training['negative_images']} negative images.
There are {training['visible_instance_ids']} visible, separately sampled vehicle arrangements in training.
These are parametric variations of eight procedural families, not independently sourced CAD models.
New traffic arrangements are sampled for each image; this is not a coherent tracking sequence.

| Family | Visible training boxes |
| --- | ---: |
{family_rows}

The larger median is not solely due to buses/trucks: audit.json includes separate passenger-car size statistics and actual range bins.
The new dataset has a broader vehicle mix and more supervision than the old pilot, so these are data-quality comparisons, not controlled model-performance comparisons.

Use data.yaml (or ../synthetic_v2.yaml) to train. vehicle_review.jpg and route_review.jpg show labeled examples.
The training entry point ../train_vehicle_v2.py can check existing real validation images with --check-real-val.
It does not automatically evaluate the already examined real test set.
No new detector has been trained or tested as part of this dataset build; improved real-world accuracy remains to be demonstrated.
Before a formal comparison, match real/synthetic training budgets, align visible/amodal conventions and reserve an appropriate final real holdout.
"""
        (ROOT / "DATASET_REPORT.md").write_text(overview)
    else:
        (ROOT / "progress_audit.json").write_text(json.dumps(report, indent=2))
        contact_sheet(records, ROOT / "vehicle_review_partial.jpg")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
