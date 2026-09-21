#!/bin/bash

# Don't forget to set to executable: sudo chown +x copy_tiles.sh
set -e
set -u

if [ -z "$1" ]; then 
    echo "Usage: $0 <text file>" >&2
    exit 2
fi

if [ "$2" = "infer" ]; then
    mkdir -p /opt/dlami/nvme/code/project/inference/tiles
    while read line; do
        aws s3 cp $line "/opt/dlami/nvme/code/project/inference/tiles/"
    done < $1
fi
if [ "$2" = "train" ]; then
    mkdir -p /opt/dlami/nvme/code/project/training/tiles
    while read line; do 
        aws s3 cp $line "/opt/dlami/nvme/code/project/training/tiles/"
    done < $1
fi
exit 0