#!/bin/bash

# Don't forget to set to executable: sudo chown +x copy_tiles.sh
set -e
set -o pipefail
set -u

if [ -z "$1" ]; then 
    echo "Usage: $0 <text file>" >&2
    exit 2
fi

mkdir -p /opt/dlami/nvme/code/project/inference/tiles
while read line; do
    aws s3 cp $line "/opt/dlami/nvme/code/Project/TrainingTiles/3in/"
done < $1
exit 0