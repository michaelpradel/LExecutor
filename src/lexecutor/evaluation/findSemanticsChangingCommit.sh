#!/bin/bash

# Arguments
if [ "$#" -ne 1 ]; then
    echo "Must pass one argument: name of project to analyze (e.g., 'flask')"
    exit
fi
project=$1

# Step 1
# Step 2
# Step 3: lexecute

for f in `find data/function_pairs/${project} -name compare.py | xargs`
do
  for i in {1..5}
  do
    timeout 30 python $f
  done
done > out_${project}_randomized