#!/bin/bash

#usage: gpkg_name row_start slice_length
#ie ./setup_inference.sh pred_5000-6000 5000 1000

mkdir -p /opt/dlami/nvme/code 
cd /opt/dlami/nvme/code

echo "retrieve code"
git clone https://github.com/indiana-gio/building-extraction.git
cd building-extraction
uv sync

echo "confirm cuda availability"
time -p echo $(uv run python -c "import torch; print(torch.cuda.is_available())")

cd ..

#copy tile reference
aws s3 cp s3://gsci-2026-building-footprint-057331986207-us-east-2-an/product/urls.csv ~/code/
#slice tile list with given indices
sh ~/code/building-extraction/ec2_scripts/parse_urls.sh ~/code/urls.csv $2 $3 > ~/code/building-extraction/ec2_scripts/infer_prod.txt

sh ~/code/building-extraction/ec2_scripts/copy_tiles.sh ~/code/building-extraction/ec2_scripts/infer_prod.txt infer
aws s3 cp s3://gsci-2026-building-footprint-057331986207-us-east-2-an/retrain/checkpoints/best.pt project/results/checkpoints/

mkdir -p /opt/dlami/nvme/code/project/inference/results

cd building-extraction
time -p uv run building_footprint_DeepLabV3.py infer

cd ../project/results/predictions
mv predicted_footprints.gpkg $1.gpkg
aws s3 cp $1.gpkg s3://gsci-2026-building-footprint-057331986207-us-east-2-an/product/
