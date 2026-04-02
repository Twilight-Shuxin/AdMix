#!/bin/bash

# File definitions
FOLDER="profiling/scheduling"
file=${FOLDER}/batch_scheduler${1}.py
dest_file=${FOLDER}/batch_scheduler.py

echo cp "$file" "$dest_file"
cp "$file" "$dest_file"