#!/bin/bash

IMG="dl_jupyter"
CON="basic_dl"

docker build -t $IMG . #> /dev/null
docker run -d -it --rm --name $CON \
    -v $(pwd):/basic_dl \
    -p 8888:8888 \
    $IMG /bin/bash -c "jupyter lab --no-browser --allow-root --port=8888 --ip=0.0.0.0"

sleep 2
echo "----------------------------------------------------------------------------"
echo "Jupyter Lab running. Visit:"
echo ""
LINK=$(docker exec -it $CON /bin/bash -c "jupyter notebook list")
echo $LINK
open "$LINK"
echo ""
echo "----------------------------------------------------------------------------"
echo "Opening conda environment"
echo "----------------------------------------------------------------------------"
docker exec -it $CON /bin/bash
echo "----------------------------------------------------------------------------"
echo "Stopping Docker container..."
echo ""
docker stop $CON > /dev/null 2>&1
docker rm $CON > /dev/null 2>&1
echo "Done"
