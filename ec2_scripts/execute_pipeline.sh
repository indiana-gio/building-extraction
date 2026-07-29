#!/bin/bash

set -e

mkdir -p /opt/dlami/nvme/code 
cd /opt/dlami/nvme/code

echo "retrieve code"
git clone https://github.com/indiana-gio/building-extraction.git
cd building-extraction
uv sync

echo "confirm cuda availability"
echo $(uv run python -c "import torch; print(torch.cuda.is_available())")

cd ..
aws s3 cp "s3://gsci-2026-building-footprint-057331986207-us-east-2-an/Project/IndyMapShapefile/" Project/IndyMapShapefile/ --recursive
aws s3 cp "s3://gsci-2026-building-footprint-057331986207-us-east-2-an/Project/RawTiles10_3inch/" Project/RawTiles10_3inch/ --recursive


