"""Stream-extract ILSVRC2012_img_train.tar (1000 per-class tars) from the NAS straight into
<dst>/<wnid>/<file>.JPEG on local disk: one sequential NAS read, no intermediate tar files.

Usage: python scripts/data/extract_imagenet_train.py <ILSVRC2012_img_train.tar> <dst train dir>
Prints progress every 50 classes and a final "DONE ..." line (~80 MiB/s from the NAS on 147).
"""
import os
import shutil
import sys
import tarfile
import time

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=False)
t0, n_cls, n_img, n_bytes = time.time(), 0, 0, 0
with tarfile.open(src, mode="r|") as outer:
    for member in outer:
        if not (member.isfile() and member.name.endswith(".tar")):
            continue
        wnid = os.path.basename(member.name)[: -len(".tar")]
        os.makedirs(os.path.join(dst, wnid))
        with tarfile.open(fileobj=outer.extractfile(member), mode="r|") as inner:
            for im in inner:
                if not im.isfile():
                    continue
                with inner.extractfile(im) as fin, open(os.path.join(dst, wnid, os.path.basename(im.name)), "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 20)
                n_img += 1
                n_bytes += im.size
        n_cls += 1
        if n_cls % 50 == 0:
            dt = time.time() - t0
            print(f"{n_cls} classes, {n_img} images, {n_bytes / 2**30:.1f} GiB, {dt / 60:.1f} min, {n_bytes / 2**20 / dt:.0f} MiB/s", flush=True)
print(f"DONE {n_cls} classes, {n_img} images, {n_bytes / 2**30:.1f} GiB in {(time.time() - t0) / 60:.1f} min", flush=True)
