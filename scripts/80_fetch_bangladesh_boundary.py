#!/usr/bin/env python
"""Download and snapshot the official geoBoundaries Bangladesh ADM0 polygon."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


API_URL = "https://www.geoboundaries.org/api/current/gbOpen/BGD/ADM0/"


def read_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "BDhighresDA-boundary-snapshot/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".metadata.json")
    if output.is_file() and not args.force:
        print(f"boundary already exists: {output}")
        return

    metadata = read_json(API_URL, 60)
    if metadata.get("boundaryISO") != "BGD" or metadata.get("boundaryType") != "ADM0":
        raise ValueError(f"unexpected geoBoundaries response: {metadata}")
    # The model grid is 0.05 degrees, so the published simplified boundary is
    # more than adequate and avoids drawing needlessly dense coast vertices.
    geometry_url = metadata.get("simplifiedGeometryGeoJSON") or metadata.get("gjDownloadURL")
    if not geometry_url:
        raise ValueError("geoBoundaries response contains no GeoJSON URL")
    boundary = read_json(geometry_url, 120)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(boundary, separators=(",", ":")) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
