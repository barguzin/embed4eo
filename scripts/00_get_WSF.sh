#!/bin/bash

# Define your target directory
SAVE_DIR="$HOME/data/WSF_Data"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

URLS=(
    "https://download.geoservice.dlr.de/WSF2019/files//WSF2019_v1_-2_4.tif"
    "https://download.geoservice.dlr.de/WSF2019/files//WSF2019_v1_0_4.tif"
)

for url in "${URLS[@]}"; do
    echo "Downloading ${url##*/} to $SAVE_DIR..."
    curl -L -O --output-dir "$SAVE_DIR" "$url"
done