#!/usr/bin/env python3
"""
Download AEF tiles for an Accra study area from Source Cooperative.

Behavior
--------
- Downloads the official annual index parquet.
- Filters tiles intersecting a user-provided bbox (defaults to Accra).
- Builds exact object URIs from index metadata when possible.
- Downloads the matching .vrt files, and optionally matching .tiff files.
- In --dry-run mode, performs a smoke test: it still downloads the index and
  downloads only the first N matched object(s) instead of the full set.

Notes
-----
- For analysis, prefer the .vrt files over the raw .tiff files.
- Source Cooperative is accessed through the Source Data Proxy endpoint using
  the AWS CLI with --endpoint-url https://data.source.coop --no-sign-request.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

DATA_PROXY = "https://data.source.coop"
S3_ROOT = "s3://tge-labs/aef/v1/annual"
INDEX_S3_CANDIDATES = [
    f"{S3_ROOT}/aef_index.parquet",
    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
]
INDEX_HTTP_CANDIDATES = [
    "https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet",
    "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
    "https://source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
]
ACCra_BBOX = (-0.396881, 5.488869, -0.021973, 5.732144)
URI_COLS = [
    "cloud_uri", "s3_uri", "uri", "url", "href", "download_url", "location"
]
PATH_COLS = [
    "path", "key", "filename", "name", "file_name", "object", "object_key"
]


class CmdError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download AEF tiles intersecting Accra")
    p.add_argument("--outdir", default="aef_accra", help="Output directory")
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2017, 2026)),
        help="Years to download (default: 2017..2025)",
    )
    p.add_argument("--west", type=float, default=ACCra_BBOX[0])
    p.add_argument("--south", type=float, default=ACCra_BBOX[1])
    p.add_argument("--east", type=float, default=ACCra_BBOX[2])
    p.add_argument("--north", type=float, default=ACCra_BBOX[3])
    p.add_argument("--download-tiffs", action="store_true", help="Also download matching TIFFs")
    p.add_argument("--keep-index", action="store_true", help="Keep local index parquet after run")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: download index and only the first matched tile(s), not the full set",
    )
    p.add_argument(
        "--probe-count",
        type=int,
        default=1,
        help="Number of matched base tiles to download in --dry-run mode (default: 1)",
    )
    return p.parse_args()


def ensure_aws() -> None:
    if shutil.which("aws") is None:
        raise SystemExit("AWS CLI not found on PATH. Install awscli or make it available in your environment.")


def aws_base_cmd() -> List[str]:
    return ["aws", "s3", "--endpoint-url", DATA_PROXY, "--no-sign-request"]


def run(cmd: List[str], *, dry_run: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=capture,
        )
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        raise CmdError(f"Command failed: {' '.join(cmd)}") from e



def is_parquet_file(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False



def cleanup_bad_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass



def download_via_http(url: str, dest: Path) -> None:
    print(f"[HTTP] {url} -> {dest}")
    try:
        with urlopen(url) as resp, dest.open("wb") as out:
            out.write(resp.read())
    except (HTTPError, URLError, Exception) as e:
        raise CmdError(f"HTTP download failed: {url}") from e



def download_index(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "aef_index.parquet"

    if index_path.exists():
        if is_parquet_file(index_path):
            print(f"[INFO] Reusing existing index: {index_path}")
            return index_path
        print(f"[WARN] Existing index is not valid parquet; removing: {index_path}")
        cleanup_bad_file(index_path)

    # Prefer the Source proxy with AWS CLI.
    for s3_uri in INDEX_S3_CANDIDATES:
        try:
            print(f"[INFO] Trying index via AWS CLI: {s3_uri}")
            cmd = aws_base_cmd() + ["cp", s3_uri, str(index_path)]
            run(cmd)
            if is_parquet_file(index_path):
                return index_path
            print("[WARN] Downloaded index is not valid parquet; removing it.")
            cleanup_bad_file(index_path)
        except Exception as e:
            print(f"[WARN] Index download failed for: {s3_uri}")
            print(f"[WARN] {e}")

    # Then try direct HTTP candidates.
    last_err: Optional[Exception] = None
    for url in INDEX_HTTP_CANDIDATES:
        try:
            print(f"[INFO] Trying index URL: {url}")
            download_via_http(url, index_path)
            if is_parquet_file(index_path):
                return index_path
            print(f"[WARN] File from {url} is not valid parquet; removing it.")
            cleanup_bad_file(index_path)
        except Exception as e:
            print(f"[WARN] Index download failed for: {url}")
            print(f"[WARN] {e}")
            last_err = e

    raise RuntimeError(f"Could not obtain a valid parquet index. Last error: {last_err}")



def load_index(index_path: Path) -> gpd.GeoDataFrame:
    print(f"[INFO] Reading index: {index_path}")
    gdf = gpd.read_parquet(index_path)

    # Normalize geometry column name if needed.
    if "geometry" not in gdf.columns:
        for candidate in ("geom", "footprint"):
            if candidate in gdf.columns:
                gdf = gdf.set_geometry(candidate)
                break
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf



def subset_index(gdf: gpd.GeoDataFrame, bbox_vals, years: List[int]) -> gpd.GeoDataFrame:
    geom = box(*bbox_vals)
    sub = gdf[gdf.geometry.intersects(geom)].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].isin(years)].copy()
    return sub



def uri_replace_suffix(uri: str, new_suffix: str) -> str:
    lower = uri.lower()
    for old in (".tiff", ".tif", ".vrt"):
        if lower.endswith(old):
            return uri[: -len(old)] + new_suffix
    return uri + new_suffix



def canonicalize_s3_uri(uri: str) -> str:
    u = uri.strip()
    if u.startswith("s3://us-west-2.opendata.source.coop/"):
        suffix = u.split("s3://us-west-2.opendata.source.coop/", 1)[1].lstrip("/")
        return f"s3://{suffix}"
    if u.startswith("s3://"):
        return u
    raise ValueError(f"Not an s3 uri: {uri}")



def normalize_to_cloud_uri(value: str, *, year: Optional[int] = None, utm_zone: Optional[str] = None) -> str:
    value = str(value).strip()

    if value.startswith("s3://"):
        return canonicalize_s3_uri(value)

    if value.startswith("http://") or value.startswith("https://"):
        needle = "/tge-labs/aef/v1/annual/"
        if needle in value:
            suffix = value.split(needle, 1)[1].lstrip("/")
            return f"{S3_ROOT}/{suffix}"
        return value

    value = value.lstrip("/")
    if value.startswith("us-west-2.opendata.source.coop/"):
        value = value.split("us-west-2.opendata.source.coop/", 1)[1].lstrip("/")
    if value.startswith("tge-labs/aef/v1/annual/"):
        suffix = value.split("tge-labs/aef/v1/annual/", 1)[1]
        return f"{S3_ROOT}/{suffix}"

    if year is not None and utm_zone is not None:
        return f"{S3_ROOT}/{year}/{utm_zone}/{Path(value).name}"

    raise ValueError(f"Could not normalize path value: {value}")



def infer_exact_object_uris(sub: gpd.GeoDataFrame) -> List[str]:
    cols = set(sub.columns)

    for col in URI_COLS:
        if col in cols:
            series = sub[col].dropna()
            if series.empty:
                continue
            uris: List[str] = []
            # Simple string values.
            if all(isinstance(v, str) for v in series.tolist()):
                try:
                    uris = [normalize_to_cloud_uri(v) for v in series.astype(str).tolist()]
                except Exception:
                    uris = []
            if uris:
                print(f"[INFO] Using URI-like column: {col}")
                return uris

    for col in PATH_COLS:
        if col in cols:
            print(f"[INFO] Using path-like column: {col}")
            uris: List[str] = []
            for _, row in sub.iterrows():
                if pd.isna(row[col]):
                    continue
                uris.append(
                    normalize_to_cloud_uri(
                        str(row[col]),
                        year=int(row["year"]) if "year" in row and pd.notna(row["year"]) else None,
                        utm_zone=str(row["utm_zone"]) if "utm_zone" in row and pd.notna(row["utm_zone"]) else None,
                    )
                )
            if uris:
                return uris

    return []



def list_zone_folder_objects(year: int, utm_zone: str) -> List[str]:
    prefix = f"{S3_ROOT}/{year}/{utm_zone}/"
    cmd = aws_base_cmd() + ["ls", prefix]
    cp = run(cmd, capture=True)

    objects: List[str] = []
    for line in cp.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if name.endswith(".vrt") or name.endswith(".tiff") or name.endswith(".tif"):
            objects.append(prefix + name)
    return objects



def build_download_list(sub: gpd.GeoDataFrame, *, include_tiffs: bool) -> List[str]:
    exact_uris = infer_exact_object_uris(sub)
    downloads: List[str] = []

    if exact_uris:
        print(f"[INFO] Found exact object paths for {len(exact_uris)} intersecting rows")
        for uri in exact_uris:
            uri = canonicalize_s3_uri(uri)
            lower = uri.lower()
            if lower.endswith(".vrt"):
                downloads.append(uri)
                if include_tiffs:
                    downloads.append(uri_replace_suffix(uri, ".tiff"))
            elif lower.endswith(".tiff") or lower.endswith(".tif"):
                if include_tiffs:
                    downloads.append(uri)
                downloads.append(uri_replace_suffix(uri, ".vrt"))
            else:
                downloads.append(uri)
                if include_tiffs:
                    downloads.append(uri + ".tiff")
                downloads.append(uri + ".vrt")
        return sorted(set(downloads))

    print("[WARN] No exact file-path column found in the index; falling back to year/UTM folders.")
    pairs = (
        sub[["year", "utm_zone"]]
        .drop_duplicates()
        .sort_values(["year", "utm_zone"])
        .itertuples(index=False)
    )
    for year, utm_zone in pairs:
        objects = list_zone_folder_objects(int(year), str(utm_zone))
        for uri in objects:
            lower = uri.lower()
            if lower.endswith(".vrt"):
                downloads.append(uri)
                if include_tiffs:
                    downloads.append(uri_replace_suffix(uri, ".tiff"))
            elif lower.endswith(".tiff") or lower.endswith(".tif"):
                if include_tiffs:
                    downloads.append(uri)
                downloads.append(uri_replace_suffix(uri, ".vrt"))
    return sorted(set(downloads))



def download_objects(uris: Iterable[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    uris = list(uris)
    if not uris:
        print("[WARN] Nothing to download.")
        return

    print(f"[INFO] Objects to download: {len(uris)}")
    for uri in uris:
        if not uri.startswith("s3://"):
            raise SystemExit(f"Unsupported URI scheme for download: {uri}")
        out_name = uri.rstrip("/").split("/")[-1]
        cmd = aws_base_cmd() + ["cp", uri, str(outdir / out_name)]
        run(cmd)



def main() -> None:
    args = parse_args()
    ensure_aws()

    outdir = Path(args.outdir).expanduser().resolve()
    bbox_vals = (args.west, args.south, args.east, args.north)

    print(f"[INFO] Output directory: {outdir}")
    print(f"[INFO] Years: {args.years}")
    print(f"[INFO] BBox: {bbox_vals}")
    if args.dry_run:
        print(f"[INFO] Dry-run smoke test enabled: will download only {args.probe_count} matched tile(s).")

    index_path = download_index(outdir)
    gdf = load_index(index_path)
    sub = subset_index(gdf, bbox_vals, args.years)

    if sub.empty:
        raise SystemExit("No intersecting tiles found for the study area and years requested.")

    print(f"[INFO] Intersecting index rows: {len(sub)}")
    if "utm_zone" in sub.columns:
        zones = sorted(set(sub["utm_zone"].astype(str).tolist()))
        print(f"[INFO] Matching UTM zones: {zones}")
    print("[INFO] First columns in index subset:", ", ".join(list(sub.columns)[:16]))

    # If we have exact paths, limit the base rows in smoke-test mode before expanding to companions.
    if args.dry_run and args.probe_count > 0:
        sub = sub.head(args.probe_count).copy()

    downloads = build_download_list(sub, include_tiffs=args.download_tiffs)

    if args.dry_run:
        print(f"[INFO] Smoke-test subset size: {len(downloads)} object(s)")

    print("[INFO] Planned download targets:")
    for uri in downloads[:10]:
        print(f"    {uri}")
    if len(downloads) > 10:
        print(f"    ... and {len(downloads) - 10} more")

    # In smoke-test mode, we still execute the probe downloads; the smaller
    # subset is what makes the run safe and fast.
    download_objects(downloads, outdir)

    if not args.keep_index and index_path.exists() and not args.dry_run:
        try:
            index_path.unlink()
            print(f"[INFO] Removed local index copy: {index_path}")
        except Exception:
            print(f"[WARN] Could not remove index copy: {index_path}")

    print("[INFO] Done. Prefer using the downloaded .vrt files for analysis.")


if __name__ == "__main__":
    main()
