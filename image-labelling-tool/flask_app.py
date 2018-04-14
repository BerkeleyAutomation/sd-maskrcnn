# The MIT License (MIT)
#
# Copyright (c) 2015 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,            
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.
import click

@click.command()
@click.option('--slic', is_flag=True, default=False, help='Use SLIC segmentation to generate initial labels')
@click.option('--readonly', is_flag=True, default=False, help='Don\'t persist changes to disk')
def run_app(slic, readonly):
    from image_labelling_tool import labelling_tool, flask_labeller

    # Specify our 3 label classes.
    # `LabelClass` parameters are: symbolic name, human readable name for UI, and RGB colour as list
    label_classes = [labelling_tool.LabelClass('tree', 'Trees', [0, 255, 192]),
                     labelling_tool.LabelClass('building', 'Buldings', [255, 128, 0]),
                     labelling_tool.LabelClass('lake', 'Lake', [0, 128, 255]),
                     ]
    if slic:
        import glob
        from matplotlib import pyplot as plt
        from skimage.segmentation import slic as slic_segment

        labelled_images = []
        for path in glob.glob('images/*.jpg'):
            print('Segmenting {0}'.format(path))
            img = plt.imread(path)
            # slic_labels = slic_segment(img, 1000, compactness=20.0)
            slic_labels = slic_segment(img, 1000, slic_zero=True) + 1

            print('Converting SLIC labels to vector labels...')
            labels = labelling_tool.ImageLabels.from_label_image(slic_labels)

            limg = labelling_tool.LabelledImageFile(path, labels)
            labelled_images.append(limg)

        print('Segmented {0} images'.format(len(labelled_images)))
    else:
        # Load in .JPG images from the 'images' directory.
        labelled_images = labelling_tool.PersistentLabelledImage.for_directory('images', image_filename_pattern='*.png',
                                                                               readonly=readonly)
        print('Loaded {0} images'.format(len(labelled_images)))


    config = {
        'tools': {
            'imageSelector': True,
            'labelClassSelector': True,
            'drawPolyLabel': True,
            'compositeLabel': True,
            'deleteLabel': True,
            'deleteConfig': {
                'typePermissions': {
                    'point': True,
                    'box': True,
                    'polygon': True,
                    'composite': True,
                    'group': True,
                }
            }
        }
    }

    flask_labeller.flask_labeller(labelled_images, label_classes, config=config)


if __name__ == '__main__':
    run_app()