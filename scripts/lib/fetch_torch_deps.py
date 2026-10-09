"""Download torch's pinned Linux x86_64 dependency wheels (nvidia-*, triton, cuda-bindings) with curl.

On 147 the Aliyun mirrors throttle HTTP/1.1 (what pip and uv speak) to ~0.24 MB/s but serve HTTP/2
(curl) at ~25 MB/s, so these large wheels are fetched here and installed from local files.

Usage: python scripts/lib/fetch_torch_deps.py <torch wheel> <dest dir>
Env:   PIP_INDEX  PEP 503 simple index (default https://mirrors.aliyun.com/pypi/simple/)
"""
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

SIMPLE = os.environ.get("PIP_INDEX", "https://mirrors.aliyun.com/pypi/simple/").rstrip("/")
PACKAGES = SIMPLE.rsplit("/simple", 1)[0] + "/packages"


def fits_cp313_x86_64(filename: str) -> bool:
    """Wheel tags installable on CPython 3.13 (non free-threaded), manylinux x86_64."""
    py, abi, plat = filename[: -len(".whl")].split("-")[-3:]
    if "manylinux" not in plat or "x86_64" not in plat:
        return False
    if abi == "abi3":
        return py.startswith("cp3") and int(py[3:]) <= 13
    return (py, abi) in {("py3", "none"), ("py2.py3", "none"), ("cp313", "cp313")}


torch_whl, dest = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(torch_whl) as zf:
    meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    meta = zf.read(meta_name).decode()

pins = []
for line in meta.splitlines():
    m = re.match(r"Requires-Dist: ([A-Za-z0-9_.\-]+)==([^\s;]+)\s*;?(.*)", line)
    if m and "extra ==" not in m.group(3) and "Linux" in m.group(3):
        pins.append((m.group(1), m.group(2)))

for name, version in pins:
    index = subprocess.run(["curl", "-fsSL", f"{SIMPLE}/{name.lower()}/"], capture_output=True, text=True, check=True).stdout
    stem = re.sub(r"[-_.]+", "_", name).lower() + "-" + version + "-"
    hrefs = [h for h in re.findall(r'href="([^"#]+\.whl)', index)
             if Path(h).name.lower().startswith(stem) and fits_cp313_x86_64(Path(h).name)]
    if not hrefs:
        raise SystemExit(f"no Linux x86_64 wheel for {name}=={version}")
    href = hrefs[-1]
    url = PACKAGES + "/" + href.split("/packages/", 1)[1]
    out = dest / Path(href).name
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(out), url], check=True)
    print(f"{name}=={version}  {out.stat().st_size / 1e6:8.1f} MB  {out.name}", flush=True)
