#!/bin/bash
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"
source myenv/bin/activate

echo "Current directory: $KGHOME"
echo "LOG_DIR: $LOG_DIR"
echo "DATA_PATH: $DATA_PATH"

