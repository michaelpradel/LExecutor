#!/bin/bash

FUNCTION_BODIES=""

declare -a PROJECTS=(
    "https://github.com/psf/black"
    "https://github.com/pallets/flask" 
    "https://github.com/pandas-dev/pandas"
    "https://github.com/scrapy/scrapy"
    "https://github.com/tensorflow/tensorflow"
)

mkdir popular_projects_snippets_dataset

# extract function bodies from projects
for project in ${PROJECTS[@]}; do
    REPO_NAME=$(echo $project | grep -o '[^/]*$')

    # delete repo in case it already exists
    rm -rf data/repos/$REPO_NAME
    # download repo
    git -C ./data/repos clone $project
    # create destination dir
    mkdir popular_projects_snippets_dataset/$REPO_NAME
    mkdir popular_projects_snippets_dataset/$REPO_NAME/functions
    mkdir popular_projects_snippets_dataset/$REPO_NAME/functions_with_invocation
    mkdir popular_projects_snippets_dataset/$REPO_NAME/bodies
    # extract function bodies
    if [ "$REPO_NAME" == "flask" ] || [ "$REPO_NAME" == "black" ]
    then
        FILES=$(find ./data/repos/$REPO_NAME/src/$REPO_NAME -type f -name "*.py")
    else
        FILES=$(find ./data/repos/$REPO_NAME/$REPO_NAME -type f -name "*.py")
    fi
	python -m lexecutor.evaluation.FunctionBodyExtractor --files $FILES --dest ./popular_projects_snippets_dataset/$REPO_NAME
    python -m lexecutor.evaluation.FunctionExtractor --files $FILES --dest ./popular_projects_snippets_dataset/$REPO_NAME
    # randomly select 200 function bodies
    FUNCTION_BODIES+="$(find ./popular_projects_snippets_dataset/$REPO_NAME/bodies -type f -name "*.py" | shuf -n 200) "
done

# save file paths to .txt files
echo $FUNCTION_BODIES | tr  ' ' '\n' > popular_projects_function_bodies_dataset.txt
sed -e 's/bodies/functions/g' -e 's/body/function/g' popular_projects_function_bodies_dataset.txt > popular_projects_functions_dataset.txt
sed -e 's/bodies/functions_with_invocation/g' popular_projects_function_bodies_dataset.txt > popular_projects_functions_with_invocation_dataset.txt

python -m lexecutor.evaluation.AddFunctionInvocation --files popular_projects_functions_dataset.txt
python -m lexecutor.evaluation.GetWrappInfo --files popular_projects_functions_dataset.txt