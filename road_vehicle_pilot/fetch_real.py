"""Download selected members of official OSDaR23 ZIPs using verified HTTP ranges."""

import io, json, urllib.request, zipfile, sys, time
from pathlib import Path

BASE = Path(__file__).parent
RAW = BASE / "real_raw"
RAW.mkdir(exist_ok=True)


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.cache = {}
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset
            if whence == 0
            else self.pos + offset if whence == 1 else self.size + offset
        )
        if self.pos < 0:
            raise ValueError("Negative offset")
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start = self.pos
        end = start + n - 1
        key = (start, end)
        if key not in self.cache:
            req = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=90) as r:
                        if r.status != 206:
                            raise RuntimeError(
                                f"Server did not honor range: {r.status}"
                            )
                        if not r.headers.get("Content-Range", "").startswith(
                            f"bytes {start}-{end}/"
                        ):
                            raise RuntimeError("Wrong byte range")
                        data = r.read(n + 1)
                    if len(data) != n:
                        raise RuntimeError("Truncated range")
                    self.cache[key] = data
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        self.pos += n
        return self.cache[key]


def catalog():
    p = RAW / "catalog.json"
    if not p.exists():
        with urllib.request.urlopen(
            "https://data.fid-move.de/api/3/action/package_show?id=osdar23", timeout=60
        ) as r:
            p.write_bytes(r.read())
    return json.loads(p.read_text())["result"]["resources"]


def archive(name):
    r = next(r for r in catalog() if r["name"] == name)
    return zipfile.ZipFile(HTTPRangeFile(r["url"]))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "1_calibration_1.1"
    with archive(name) as z:
        names = z.namelist()
        print(
            json.dumps(
                {
                    "sequence": name,
                    "members": len(names),
                    "first_members": names[:18],
                    "json": [n for n in names if n.endswith(".json")],
                },
                indent=2,
            )
        )
        for n in names:
            if n.endswith(".json"):
                p = RAW / name / Path(n).name
                p.parent.mkdir(exist_ok=True)
                p.write_bytes(z.read(n))
                print("SAVED", p)
