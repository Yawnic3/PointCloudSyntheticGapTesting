"""Fetch published OpenMMLab PointPillars weights, reference code and demo."""

import hashlib
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "pretrained_pointpillars"
ROOT = "https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v1.4.0/"
FILES = [
    "LICENSE",
    "configs/_base_/models/pointpillars_hv_secfpn_kitti.py",
    "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
    "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py",
    "configs/pointpillars/metafile.yml",
    "mmdet3d/models/voxel_encoders/pillar_encoder.py",
    "mmdet3d/models/voxel_encoders/utils.py",
    "mmdet3d/models/backbones/second.py",
    "mmdet3d/models/necks/second_fpn.py",
    "mmdet3d/models/middle_encoders/pillar_scatter.py",
    "mmdet3d/models/dense_heads/anchor3d_head.py",
    "mmdet3d/models/task_modules/coders/delta_xyzwhlr_bbox_coder.py",
    "mmdet3d/models/task_modules/anchor/anchor_3d_generator.py",
    "demo/data/kitti/000008.bin",
]
WEIGHTS = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    sources = [(ROOT + name, OUT / "official" / name) for name in FILES]
    sources.append((WEIGHTS, OUT / "weights" / WEIGHTS.rsplit("/", 1)[-1]))
    for url, path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".pth":
            assert digest.startswith(
                "d42d15ed"
            ), "Published checkpoint hash prefix mismatch"
        rows.append(
            {
                "url": url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        print(
            "OFFICIAL_POINTPILLARS_FETCH",
            path.name,
            path.stat().st_size,
            digest,
            flush=True,
        )
    (OUT / "download_manifest.json").write_text(
        json.dumps({"official_reference_tag": "v1.4.0", "files": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
