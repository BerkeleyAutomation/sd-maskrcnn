#!/bin/bash

# Install maskrcnn submodule
cd maskrcnn && python setup.py install && cd ..

# Install main module, with generation if requested
if [ "$#" == "1" ] && [ "$1" == "generation" ]
then
    echo -n "Installing generation requirements"
    pip install .[generation]
else
    pip install .
fi

# Download pretrained model if requested
shopt -s nocasematch
unset response
while [[ ! $response =~ (y|yes|n|no) ]]; do
    echo -n "Download Pre-trained Model to models (150 MB)? (y/n) > "
    read response
done

case "$response" in
    [yY][eE][sS]|[yY]) 
        mkdir -p models
        wget -v -O models/sd_maskrcnn.h5 https://berkeley.box.com/shared/static/obj0b2o589gc1odr2jwkx4qjbep11t0o.h5
        ;;
    *)
        ;;
esac
echo "Done"
