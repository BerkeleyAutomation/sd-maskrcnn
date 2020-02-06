#!/bin/bash

""" 
Copyright ©2019. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk

Installs sd_maskrcnn repo and other dependencies, as well as optional models and datasets.
"""

# Install maskrcnn submodule
git submodule update --init
cd maskrcnn 
python3 setup.py install
cd ..

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
    echo -n "Download Pre-trained Model to models/ (150 MB)? (y/n) > "
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

if [ "$#" == "1" ] && [ "$1" == "generation" ]
then
    shopt -s nocasematch
    unset response
    while [[ ! $response =~ (y|yes|n|no) ]]; do
        echo -n "Download ycb object models to datasets/objects/meshes/ycb/ (45 MB)? (y/n) > "
        read response
    done

    case "$response" in
        [yY][eE][sS]|[yY]) 
            python tools/download_ycb_dataset.py
            ;;
        *)
            ;;
    esac
fi
echo "Done"
