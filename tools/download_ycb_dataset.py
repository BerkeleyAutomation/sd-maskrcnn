# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
from urllib.request import urlopen, Request
import shutil

output_directory = "./datasets/objects/meshes/ycb"
base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_to_download = [
    "001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box",	
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box",
    "009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry", 
    "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange",
    "018_plum", "019_pitcher_base", "021_bleach_cleanser", "022_windex_bottle", "024_bowl",
    "026_sponge", "027_skillet", "029_plate", "030_fork", "031_spoon", "032_knife",
    "033_spatula", "035_power_drill", "036_wood_block", "037_scissors", "038_padlock",
    "040_large_marker", "041_small_marker", "042_adjustable_wrench", "043_phillips_screwdriver", 
    "044_flat_screwdriver", "048_hammer", "050_medium_clamp", "051_large_clamp", 
    "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball", "055_baseball", "056_tennis_ball", 
    "057_racquetball", "058_golf_ball", "059_chain", "061_foam_brick", "062_dice", "063-a_marbles", 
    "065-a_cups", "070-a_colored_wood_blocks", "071_nine_hole_peg_test", "072-a_toy_airplane", 
    "073-a_lego_duplo", "076_timer", "077_rubiks_cube"
]

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def download_file(url, filename):
    u = urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.get("Content-Length"))
    print("Downloading: {} ({:.2f} MB)".format(filename, file_size/1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        sys.stdout.write('\r')
        status = "{:.1f} MB / {:.1f} MB  [{:3.2f}%]".format(file_size_dl/1000000.0, file_size/1000000.0, file_size_dl * 100. / file_size)
        sys.stdout.write(status)
        sys.stdout.flush()
    sys.stdout.write('\n')
    f.close()

def tgz_url(obj):
    return base_url + "berkeley/{obj}/{obj}_berkeley_meshes.tgz".format(obj=obj)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)
    obj_bn = os.path.splitext(filename)[0]
    mesh_fn = os.path.join(obj_bn, 'poisson', 'nontextured.stl')
    os.rename(mesh_fn, obj_bn + '.stl')
    shutil.rmtree(obj_bn)

def check_url(url):
    try:
        request = Request(url)
        request.get_method = lambda : 'HEAD'
        response = urlopen(request)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":

    for obj in objects_to_download:
        url = tgz_url(obj)
        if not check_url(url):
            continue
        filename = "{path}/{object}.tgz".format(path=output_directory,
                                                            object=obj)
        download_file(url, filename)
        extract_tgz(filename, output_directory)
