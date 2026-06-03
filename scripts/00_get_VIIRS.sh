#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/data/tmp}"
URL="${URL:-https://data.worldpop.org/GIS/Covariates/Global_2015_2030/GHA/VIIRS/v1/fvf//gha_viirs_fvf_2019_100m_v1.tif}"
OUT_FILE="$OUT_DIR/gha_viirs_fvf_2019_100m_v1.tif"

mkdir -p "$OUT_DIR"

echo "Downloading VIIRS FVF 2019 100 m covariate to:"
echo "  $OUT_FILE"

curl -L "$URL" -o "$OUT_FILE"

echo "Finished downloading VIIRS:"
echo "  $OUT_FILE"
