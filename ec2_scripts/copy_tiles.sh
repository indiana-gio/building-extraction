#!/bin/bash

# Don't forget to set to executable: sudo chown +x copy_tiles.sh
set -e
set -o pipefail
set -u

if [ -z "$1" ]; then 
    echo "Usage: $0 <text file>" >&2
    exit 2
fi

mkdir -p /opt/dlami/nvme/code/Project/TrainingTiles/3in
while read line; do
    aws s3 cp "s3://gisimageryingov/imageryoptimized/statewide/2025/SPE/03in/$line.tif" "/opt/dlami/nvme/code/Project/TrainingTiles/3in/$line.tif"
done < $1
exit 0