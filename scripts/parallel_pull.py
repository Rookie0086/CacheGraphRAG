#!/usr/bin/env python3
"""Parallel image puller for throttled registry mirrors.

Downloads all blobs of an image via concurrent HTTP range requests,
verifies sha256 digests, assembles a docker-save-compatible archive,
and loads it into the local docker daemon.

Usage: python3 parallel_pull.py <repo:tag> [<repo:tag> ...]
"""
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request

MIRRORS = [
    "https://docker.1panel.live",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
]
UA = "Docker-Client/24.0.0"
SEG = 4 * 1024 * 1024      # 4MB per segment
THREADS = 16               # concurrent segment downloads
SMALL = 8 * 1024 * 1024    # blobs under 8MB: single request
RETRIES = 4

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def http_get(url, headers=None, timeout=60):
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout)


def mirror_get(path, accept, mirrors=MIRRORS):
    """Try each mirror (2 rounds with delay); return (response_bytes, mirror_used)."""
    import time
    last_err = None
    for rnd in range(3):
        for m in mirrors:
            try:
                with http_get(m + path, {"Accept": accept}) as r:
                    return r.read(), m
            except urllib.error.HTTPError as e:
                if e.code in (404, 403):
                    last_err = e  # definitive: this mirror lacks the object
                    continue      # try next mirror immediately
                last_err = e
            except Exception as e:
                last_err = e  # network hiccup -> retry later rounds
        if rnd < 2:
            time.sleep(2)
    raise RuntimeError(f"all mirrors failed for {path}: {last_err}")


def resolve_manifest(repo, tag):
    """Return (single_arch_manifest_dict, mirror_list_preferring_working_one)."""
    data, m = mirror_get(f"/v2/{repo}/manifests/{tag}",
                         "application/vnd.docker.distribution.manifest.list.v2+json,"
                         "application/vnd.docker.distribution.manifest.v2+json,"
                         "application/vnd.oci.image.manifest.v1+json,"
                         "application/vnd.oci.image.index.v1+json")
    man = json.loads(data)
    if "manifests" in man:  # index / manifest list -> pick arm64
        for entry in man["manifests"]:
            p = entry.get("platform", {})
            if p.get("architecture") == "arm64":
                digest = entry["digest"]
                data, m = mirror_get(f"/v2/{repo}/manifests/{digest}",
                                     "application/vnd.docker.distribution.manifest.v2+json,"
                                     "application/vnd.oci.image.manifest.v1+json")
                return json.loads(data), [m] + [x for x in MIRRORS if x != m]
        raise RuntimeError(f"no arm64 manifest for {repo}:{tag}")
    return man, [m] + [x for x in MIRRORS if x != m]


def download_segment(url, start, end, dest, mirrors, sem, stats):
    """Download one byte range to dest file at offset start."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(RETRIES):
        for m in mirrors:
            try:
                with http_get(m + url, headers, timeout=300) as r:
                    buf = r.read()
                if len(buf) != end - start + 1:
                    raise IOError(f"short read {len(buf)} != {end-start+1}")
                with sem:
                    with open(dest, "r+b") as f:
                        f.seek(start)
                        f.write(buf)
                    stats["done"] += len(buf)
                    total = stats["size"]
                    if stats["done"] - stats["last_report"] > 8 * 1024 * 1024:
                        stats["last_report"] = stats["done"]
                        log(f"    {stats['done']/1048576:.0f}/{total/1048576:.0f} MB")
                return
            except Exception:
                continue
    raise RuntimeError(f"segment {start}-{end} failed after retries")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def download_blob(repo, digest, size, dest, mirrors):
    """Download a blob (parallel range segments for large blobs), verify sha256."""
    if os.path.exists(dest) and _sha256_file(dest) == digest:
        log(f"  [skip] {digest[:19]} already downloaded")
        return
    url = f"/v2/{repo}/blobs/{digest}"
    with open(dest, "wb") as f:
        f.truncate(size)

    if size == 0:
        return

    if size <= SMALL:
        data, _ = mirror_get(url, "*/*", mirrors)
        if len(data) != size:
            raise IOError(f"blob size mismatch: {len(data)} != {size}")
        with open(dest, "wb") as f:
            f.write(data)
    else:
        sem = threading.Semaphore(THREADS)
        stats = {"done": 0, "size": size, "last_report": 0}
        threads = []
        errors = []
        for start in range(0, size, SEG):
            end = min(start + SEG, size) - 1

            def worker(url=url, start=start, end=end):
                try:
                    download_segment(url, start, end, dest, mirrors, sem, stats)
                except Exception as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError(f"blob {digest} had {len(errors)} failed segments: {errors[0]}")

    actual = _sha256_file(dest)
    if actual != digest:
        os.remove(dest)
        raise RuntimeError(f"sha256 mismatch for {digest}: got {actual}")


def pull_image(repo, tag, workdir):
    log(f"\n==> Pulling {repo}:{tag}")
    man, mirrors = resolve_manifest(repo, tag)
    cfg_digest = man["config"]["digest"]
    layers = man["layers"]
    total = sum(l["size"] for l in layers)
    log(f"  arm64 manifest ok: {len(layers)} layers, {total/1048576:.1f} MB")

    blobs_dir = os.path.join(workdir, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)

    # config blob
    cfg_path = os.path.join(blobs_dir, cfg_digest.replace(":", "_") + ".json")
    download_blob(repo, cfg_digest, man["config"]["size"], cfg_path, mirrors)

    # layer blobs
    layer_paths = []
    for i, l in enumerate(layers):
        p = os.path.join(blobs_dir, l["digest"].replace(":", "_") + ".blob")
        log(f"  layer {i+1}/{len(layers)}: {l['size']/1048576:.1f} MB {l['digest'][:19]}")
        download_blob(repo, l["digest"], l["size"], p, mirrors)
        layer_paths.append((p, l["digest"]))

    # assemble docker-save archive
    log("  assembling docker archive ...")
    archive = os.path.join(workdir, f"{repo.replace('/', '_')}_{tag}.tar")
    with tarfile.open(archive, "w") as tar:
        # config
        arcname = cfg_digest.replace(":", "") + ".json"
        tar.add(cfg_path, arcname=arcname)

        layer_tar_names = []
        for p, digest in layer_paths:
            arc = digest.replace(":", "") + ".tar"
            with open(p, "rb") as f:
                head = f.read(2)
            if head == b"\x1f\x8b":  # gzip
                with gzip.open(p, "rb") as gz, open(p + ".tar", "wb") as out:
                    shutil.copyfileobj(gz, out)
                tar.add(p + ".tar", arcname=arc)
                os.remove(p + ".tar")
            else:
                tar.add(p, arcname=arc)
            layer_tar_names.append(arc)

        manifest = [{
            "Config": arcname,
            "RepoTags": [f"{repo}:{tag}"],
            "Layers": layer_tar_names,
        }]
        mdata = json.dumps(manifest).encode()
        ti = tarfile.TarInfo("manifest.json")
        ti.size = len(mdata)
        tar.addfile(ti, io.BytesIO(mdata))

    log(f"  loading into docker: {os.path.basename(archive)}")
    r = subprocess.run(["docker", "load", "-i", archive], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  docker load FAILED:\n{r.stdout}\n{r.stderr}")
        return False
    log(f"  docker load OK: {r.stdout.strip()}")
    os.remove(archive)
    return True


def main():
    images = sys.argv[1:]
    if not images:
        print(__doc__)
        sys.exit(1)
    workdir = os.path.expanduser("~/docker_image_cache")
    os.makedirs(workdir, exist_ok=True)
    ok, fail = [], []
    for it in images:
        repo, tag = it.rsplit(":", 1)
        try:
            if pull_image(repo, tag, workdir):
                ok.append(it)
            else:
                fail.append(it)
        except Exception as e:
            log(f"  ERROR pulling {it}: {e}")
            fail.append(it)
    print(f"\nDone. OK: {ok}  FAILED: {fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
