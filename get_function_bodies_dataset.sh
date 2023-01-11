#!/bin/bash

FUNCTION_BODIES=""

declare -a PROJECTS=(
    "https://github.com/psf/black"
    "https://github.com/pallets/flask" 
    "https://github.com/pandas-dev/pandas"
    "https://github.com/scrapy/scrapy"
    "https://github.com/tensorflow/tensorflow"
)

mkdir function_bodies_dataset

# extract function bodies from projects
for project in ${PROJECTS[@]}; do
    REPO_NAME=$(echo $project | grep -o '[^/]*$')

    # delete repo in case it already exists
    rm -rf data/repos/$REPO_NAME
    # download repo
    git -C ./data/repos clone $project
    # create destination dir
    mkdir function_bodies_dataset/$REPO_NAME
    # extract function bodies
    if [ "$REPO_NAME" == "flask" ] || [ "$REPO_NAME" == "black" ]
    then
        FILES=$(find ./data/repos/$REPO_NAME/src/$REPO_NAME -type f -name "*.py")
    else
        FILES=$(find ./data/repos/$REPO_NAME/$REPO_NAME -type f -name "*.py")
    fi
	python -m lexecutor.FunctionBodyExtractor --files $FILES --dest ./function_bodies_dataset/$REPO_NAME
    # randomly select 200 function bodies
    FUNCTION_BODIES+="$(find ./function_bodies_dataset/$REPO_NAME -type f -name "*.py" | shuf -n 200) "
done

echo $FUNCTION_BODIES | tr  ' ' '\n' > function_bodies_dataset.txt