#!/usr/bin/env bash
FOLDER=$(dirname $0)
FOLDER=$(realpath $FOLDER)
rm -rf $FOLDER/dist
python -m build --wheel . 
