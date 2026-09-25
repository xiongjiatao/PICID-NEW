"""Download the NASA-listed archive and extract only the four study sources.

This CPU-only data preparation step never launches model fitting. Status is saved
after every 64 MiB and each extracted file. Raw archive names/hashes are retained.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import requests

URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
SOURCE = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
NAMES = {
    "N-CMAPSS_DS01-005.h5": "N-CMAPSS_DS01.h5",
    "N-CMAPSS_DS01.h5": "N-CMAPSS_DS01.h5",
    "N-CMAPSS_DS04.h5": "N-CMAPSS_DS04.h5",
    "N-CMAPSS_DS05.h5": "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS07.h5": "N-CMAPSS_DS07.h5",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    archive = args.data_root / "NASA_N-CMAPSS.zip"
    partial = archive.with_suffix(".zip.part")
    state = {
        "url": URL,
        "source": SOURCE,
        "command": sys.argv,
        "python": sys.executable,
        "cwd": os.getcwd(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_gpu": None,
        "pid": os.getpid(),
        "status": "STARTING",
        "started": time.time(),
        "files": [],
    }

    def save():
        state["updated"] = time.time()
        temp = args.status.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(args.status)

    try:
        save()
        with requests.Session() as session:
            head = session.head(URL, timeout=30)
            head.raise_for_status()
            size = int(head.headers["Content-Length"])
            etag = head.headers.get("ETag")
            state.update(total_bytes=size, etag=etag)
            if shutil.disk_usage(args.data_root).free < 3 * size:
                raise RuntimeError(
                    "Insufficient disk headroom for archive and selected raw sources"
                )
            if not archive.exists():
                offset = partial.stat().st_size if partial.exists() else 0
                meta = partial.with_suffix(".metadata.json")
                if offset and (
                    not meta.exists()
                    or json.loads(meta.read_text()).get("etag") != etag
                ):
                    raise RuntimeError(
                        "Partial download provenance differs; refusing unsafe resume"
                    )
                meta.write_text(json.dumps({"url": URL, "etag": etag, "size": size}))
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session.get(
                    URL, headers=headers, stream=True, timeout=(30, 120)
                ) as response:
                    response.raise_for_status()
                    if offset and (
                        response.status_code != 206
                        or not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        )
                    ):
                        raise RuntimeError(
                            "Server did not honor requested resume offset"
                        )
                    state.update(status="DOWNLOADING", downloaded_bytes=offset)
                    save()
                    with partial.open("ab" if offset else "wb") as stream:
                        checkpoint = offset
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if offset - checkpoint >= 64 * 1024 * 1024:
                                state["downloaded_bytes"] = offset
                                save()
                                checkpoint = offset
                if partial.stat().st_size != size:
                    raise RuntimeError(
                        "Downloaded archive size differs from remote size"
                    )
                partial.replace(archive)
            elif archive.stat().st_size != size:
                raise RuntimeError("Existing archive size mismatch")
        state.update(status="EXTRACTING", downloaded_bytes=size)
        save()
        target_dir = args.data_root / "N-CMAPSS"
        target_dir.mkdir(exist_ok=True)
        seen = set()
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                original = Path(member.filename).name
                if original not in NAMES:
                    continue
                target_name = NAMES[original]
                if target_name in seen:
                    raise RuntimeError(f"Ambiguous duplicate source: {target_name}")
                seen.add(target_name)
                target = target_dir / target_name
                if target.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite existing source: {target}"
                    )
                temp = target.with_suffix(".h5.part")
                digest = hashlib.sha256()
                with source.open(member) as inp, temp.open("wb") as out:
                    while chunk := inp.read(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                # Reading to EOF validates ZIP CRC as well as recording SHA256.
                temp.replace(target)
                state["files"].append(
                    {
                        "archive_member": member.filename,
                        "local_name": target_name,
                        "bytes": target.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
                save()
        if seen != set(NAMES.values()):
            raise RuntimeError(
                f"Archive does not contain expected sources: {set(NAMES.values()) - seen}"
            )
        state["status"] = "DATA_READY_PROTOCOL_AUDIT_STILL_REQUIRED"
        save()
    except Exception as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
