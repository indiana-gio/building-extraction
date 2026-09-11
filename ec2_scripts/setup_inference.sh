#!/bin/bash

mkdir -p /opt/dlami/nvme/code 
cd /opt/dlami/nvme/code

echo "retrieve code"
git clone https://github.com/indiana-gio/building-extraction.git
cd building-extraction
uv sync

echo "confirm cuda availability"
time -p echo $(uv run python -c "import torch; print(torch.cuda.is_available())")

cd ..

sh ~/building-extraction/ec2_scripts/copy_tiles.sh ~/building-extraction/ec2_scripts/one-off.txt
aws s3 cp s3://gsci-2026-building-footprint-057331986207-us-east-2-an/CustomModel/Results_Large_Run/checkpoints/best.pt project/inference/weights/

mkdir -p /opt/dlami/nvme/code/project/inference/results

cd building-extraction
time -p uv run building_footprint_DeepLabV3.py infer

cd ../project/inference
mv weights/*.!(pt) results/
