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



import base64, json, sys, six

from ipywidgets import widgets

from IPython.utils.traitlets import Unicode, Integer, List, Dict

from . import labelling_tool





class ImageLabellingTool (widgets.DOMWidget):
    _view_name = Unicode('ImageLabellingToolView', sync=True)
    _view_module = Unicode('image-labelling-tool', sync=True)

    label_classes = List(sync=True)

    tool_width_ = Integer(sync=True)
    tool_height_ = Integer(sync=True)

    images_ = List(sync=True)
    initial_image_index_ = Integer(sync=True)

    labelling_tool_config_ = Dict(sync=True)



    def __init__(self, labelled_images=None, label_classes=None, tool_width=1040, tool_height=585,
                 labelling_tool_config=None, **kwargs):
        """

        :type labelled_images: AbstractLabelledImage
        :param labelled_images: a list of images to label

        :type label_classes: [LabelClass]
        :param label_classes: list of label classes available to the user

        :type tool_width: int
        :param tool_width: width of tool in pixels

        :type tool_height: int
        :param tool_height: height of tool in pixels

        :param kwargs: kwargs passed to DOMWidget constructor
        """
        if label_classes is None:
            label_classes = []

        label_classes = [{'name': cls.name, 'human_name': cls.human_name, 'colour': cls.colour}   for cls in label_classes]

        if labelled_images is None:
            labelled_images = []

        if labelling_tool_config is None:
            labelling_tool_config = {}

        image_ids = [str(i)   for i in range(len(labelled_images))]
        self.__images = {image_id: img   for image_id, img in zip(image_ids, labelled_images)}
        self.__changing = False

        image_descriptors = []
        for image_id, img in zip(image_ids, labelled_images):
            image_descriptors.append(labelling_tool.image_descriptor(image_id=image_id))


        super(ImageLabellingTool, self).__init__(tool_width_=tool_width, tool_height_=tool_height,
                                                 images_=image_descriptors,
                                                 initial_image_index_=0,
                                                 label_classes=label_classes,
                                                 labelling_tool_config_=labelling_tool_config, **kwargs)

        self.on_msg(self._on_msg_recv)

        self.label_data = labelled_images[0].labels_json


    def _on_msg_recv(self, _, msg, *args):
        msg_type = msg.get('msg_type', '')
        if msg_type == 'get_labels':
            try:
                image_id = str(msg.get('image_id', '0'))
            except ValueError:
                image_id = '0'

            load_labels_msg = {}

            image = self.__images[image_id]
            data, mimetype, width, height = image.data_and_mime_type_and_size()

            data_b64 = base64.b64encode(data)

            if sys.version_info[0] == 3:
                data_b64 = data_b64.decode('us-ascii')

            labels_json, complete = image.get_label_data_for_tool()

            self.label_data = labels_json

            msg_label_header = {
                'image_id': image_id,
                'labels': labels_json,
                'complete': complete
            }
            msg_image = {
                'image_id': image_id,
                'img_url': 'data:{0};base64,'.format(mimetype) + data_b64,
                'width': width,
                'height': height,
            }
            self.send({
                'msg_type': 'load_labels',
                'label_header': msg_label_header,
                'image': msg_image,
            })
        elif msg_type == 'update_labels':
            label_header = msg.get('label_header')
            if label_header is not None:
                image_id = label_header['image_id']
                complete = label_header['complete']
                labels = label_header['labels']
                self.__images[image_id].set_label_data_from_tool(labels, complete)
                print('Received changes for image {0}; {1} labels'.format(image_id, len(labels)))



_LABELLING_TOOL_JS_URLS = labelling_tool.js_file_urls("image_labelling_tool/static/labelling_tool")
_LABELLING_TOOL_JS_REFS = ', '.join(['"{}"'.format(url) for url in _LABELLING_TOOL_JS_URLS])

def _lt_deps_shim():
    # Regrettably Jupyter uses require.js which burdens us with defining dependencies.
    # The files are in the order they should be imported, so just make sure each one
    # depends on the previous file.
    # Duplicating all the dependencies that are defined in the Typescript files takes time
    # and I have better things to do with my life. I hate web development. I hate Javascript.
    shim = {}
    # files = [f.replace('.js', '') for f in labelling_tool.LABELLING_TOOL_JS_FILES]
    files = _LABELLING_TOOL_JS_URLS
    shim[files[0]] = dict(exports='labelling_tool')
    for prev, cur in zip(files[:-1], files[1:]):
        shim[cur] = dict(deps=[prev], exports='labelling_tool')
    return json.dumps(shim)


LABELLING_TOOL_JUPYTER_JS = """
console.log("Run me");
requirejs.config({
    shim: {{<<SHIM>>}}
});

define('image-labelling-tool',
       ["jupyter-js-widgets",
        "image_labelling_tool/static/d3.min.js",
        "image_labelling_tool/static/json2.js",
        "image_labelling_tool/static/polyk.js",
        {{<<REFS>>}}],
       function(widget, manager){
    /*
    Labeling tool view; links to the server side data structures
     */
    var ImageLabellingToolView = widget.DOMWidgetView.extend({
        render: function() {
            var self = this;

            // Register a custom IPython widget message handler for receiving messages from the Kernel
            this.model.on('msg:custom', this._on_custom_msg, this);


            // Get label classes, tool dimensions, and image ID set and initial image ID from the kernel
            var label_classes = self.model.get("label_classes");
            var tool_width = self.model.get("tool_width_");
            var tool_height = self.model.get("tool_height_");
            var images = self.model.get('images_');
            var initial_image_index = self.model.get('initial_image_index_');
            var config = self.model.get('labelling_tool_config_');

            console.log("Labelling tool config:");
            console.log(config);


            // Callback function to allow the labelling tool to request an image
            var get_labels = function(image_id_str) {
                // Send a 'request_image_descriptor' message to the kernel requesting the
                // image identified by `image_id_str`
                self.send({msg_type: 'get_labels', image_id: image_id_str});
            };

            // Callback function to allow the labelling tool to send modified label data to the kernel
            var update_labels = function(label_header) {
                // Send a 'label_header' message to the kernel, along with modified label data
                self.send({msg_type: 'update_labels', label_header: label_header});
            };

            // Create the labelling tool
            // Place it into the widget element (`this.$el`).
            // Also give it the label classes, tool dimensions, image ID set, initial image ID and the callbacks above
            self._labeling_tool = new labelling_tool.LabellingTool(this.$el, label_classes, tool_width, tool_height,
                                                                   images, initial_image_index,
                                                                   get_labels, update_labels, null,
                                                                   config);
        },


        _on_custom_msg: function(msg) {
            // Received a custom message from the kernel
            if (msg.msg_type === "load_labels") {
                // 'load_labels' message
                var label_header = msg.label_header;
                var image = msg.image;
                // Send labels to labelling tool
                this._labeling_tool.loadLabels(label_header, image);
            }
        }
    });

    // Register the ImageLabelingToolView with the IPython widget manager.
//     manager.WidgetManager.register_widget_view('ImageLabellingToolView', ImageLabellingToolView);
    console.log("Defined ImageLabellingToolView");

    return {
        'ImageLabellingToolView': ImageLabellingToolView
    };
});
""".replace('{{<<REFS>>}}', _LABELLING_TOOL_JS_REFS)\
    .replace('{{<<SHIM>>}}', _lt_deps_shim())

