#!/usr/bin/env bash
set -euo pipefail

# Inputs
BBOX_GPKG="${BBOX_GPKG:-$HOME/data/aux/bbox_accra_dissolve.gpkg}"
OUT_DIR="${OUT_DIR:-$HOME/data/ESA_cover_2020}"
DOWNLOAD_SCRIPT="${DOWNLOAD_SCRIPT:-$HOME/Documents/GitHub/embed4eo/scripts/download_ESA.py}"
YEAR=2020
ENV_NAME="diss"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BBOX_GPKG" ]]; then
  echo "ERROR: Bounding box GeoPackage not found:"
  echo "  $BBOX_GPKG"
  exit 1
fi

if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
  echo "ERROR: ESA download script not found:"
  echo "  $DOWNLOAD_SCRIPT"
  exit 1
fi

echo "Using mamba environment:"
echo "  $ENV_NAME"
echo

BOUNDS=$(
mamba run -n "$ENV_NAME" python - <<PY
import geopandas as gpd
from pathlib import Path

bbox_path = Path("$BBOX_GPKG").expanduser()

gdf = gpd.read_file(bbox_path)

if gdf.empty:
    raise ValueError(f"No features found in {bbox_path}")

if gdf.crs is None:
    raise ValueError(
        f"{bbox_path} has no CRS. Please define the CRS before running this script."
    )

gdf = gdf.to_crs(4326)

xmin, ymin, xmax, ymax = gdf.total_bounds
print(xmin, ymin, xmax, ymax)
PY
)

echo "Using bounding box:"
echo "  $BOUNDS"
echo

echo "Saving ESA WorldCover $YEAR tiles to:"
echo "  $OUT_DIR"
echo

mamba run -n "$ENV_NAME" python "$DOWNLOAD_SCRIPT" \
  --output "$OUT_DIR" \
  --bounds $BOUNDS \
  --year "$YEAR"


  ## run it by chmod +x scripts/00_get_ESA.sh
  ## then ./scripts/00_get_ESA.sh
  