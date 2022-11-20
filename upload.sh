#!/usr/bin/env bash
FOLDER=$(dirname $0)
FOLDER=$(realpath $FOLDER)
python -m twine upload --verbose --repository testpypi $FOLDER/dist/*
