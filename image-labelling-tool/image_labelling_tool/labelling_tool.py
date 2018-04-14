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


import mimetypes, json, os, glob, io, math, six, traceback

import numpy as np

import random

from PIL import Image, ImageDraw

from skimage import img_as_float
from skimage import transform
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.util import pad
from skimage.measure import find_contours


LABELLING_TOOL_JS_FILES = [
    'math_primitives.js',
    'object_id_table.js',
    'label_class.js',
    'abstract_label.js',
    'abstract_tool.js',
    'select_tools.js',
    'point_label.js',
    'box_label.js',
    'composite_label.js',
    'polygonal_label.js',
    'group_label.js',
    'main_tool.js',
    'root_label_view.js',
]

def js_file_urls(url_prefix):
    if not url_prefix.endswith('/'):
        url_prefix = url_prefix + '/'
    return ['{}{}'.format(url_prefix, filename) for filename in LABELLING_TOOL_JS_FILES]

class LabelClass (object):
    def __init__(self, name, human_name, colour):
        """
        Label class constructor
        
        :param name: identifier class name 
        :param human_name: human readable name
        :param colour: colour as a tuple or list e.g. [255, 0, 0] for red
        """
        self.name = name
        self.human_name = human_name
        colour = list(colour)
        if len(colour) != 3:
            raise TypeError('colour must be a tuple or list of length 3')
        self.colour = colour


    def to_json(self):
        return {'name': self.name, 'human_name': self.human_name, 'colour': self.colour}


def label_class(name, human_name, rgb):
    return {'name': name,
            'human_name': human_name,
            'colour': rgb}

def image_descriptor(image_id, url=None, width=None, height=None):
    return {'image_id': str(image_id),
            'img_url': str(url) if url is not None else None,
            'width': width,
            'height': height,}


def _next_wrapped_array(xs):
    return np.append(xs[1:], xs[:1], axis=0)

def _prev_wrapped_array(xs):
    return np.append(xs[-1:], xs[:-1], axis=0)

def _simplify_contour(cs):
    degenerate_verts = (cs == _next_wrapped_array(cs)).all(axis=1)
    while degenerate_verts.any():
        cs = cs[~degenerate_verts,:]
        degenerate_verts = (cs == _next_wrapped_array(cs)).all(axis=1)

    if cs.shape[0] > 0:
        # Degenerate eges
        edges = (_next_wrapped_array(cs) - cs)
        edges = edges / np.sqrt((edges**2).sum(axis=1))[:,None]
        degenerate_edges = (_prev_wrapped_array(edges) * edges).sum(axis=1) > (1.0 - 1.0e-6)
        cs = cs[~degenerate_edges,:]

        if cs.shape[0] > 0:
            return cs
    return None



_LABEL_CLASS_REGISTRY = {}


def label_cls(cls):
    json_label_type = cls.__json_type_name__
    _LABEL_CLASS_REGISTRY[json_label_type] = cls
    return cls



class LabelContext (object):
    def __init__(self, point_radius=0.0):
        self.point_radius = point_radius


class AbstractLabel (object):
    __json_type_name__ = None

    def __init__(self, object_id=None, classification=None):
        """
        Constructor

        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        self.object_id = object_id
        self.classification = classification

    @property
    def dependencies(self):
        return []

    def flatten(self):
        yield self

    def bounding_box(self, ctx=None):
        raise NotImplementedError('Abstract')

    def _warp(self, xform_fn, object_table):
        raise NotImplementedError('Abstract')

    def warped(self, xform_fn, object_table):
        w = self._warp(xform_fn, object_table)
        object_table.register(w)
        return w

    def _render_mask(self, img, fill, dx=0.0, dy=0.0, ctx=None):
        raise NotImplementedError('Abstract')

    def render_mask(self, width, height, fill, dx=0.0, dy=0.0, ctx=None):
        img = Image.new('L', (width, height), 0)
        self._render_mask(img, fill, dx, dy, ctx)
        return np.array(img)

    def to_json(self):
        return dict(label_type=self.__json_type_name__, object_id=self.object_id, label_class=self.classification)

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        raise NotImplementedError('Abstract')


    @staticmethod
    def from_json(label_json, object_table):
        label_type = label_json['label_type']
        cls = _LABEL_CLASS_REGISTRY.get(label_type)
        if cls is None:
            raise TypeError('Unknown label type {0}'.format(label_type))
        label = cls.new_instance_from_json(label_json, object_table)
        object_table.register(label)
        return label


@label_cls
class PointLabel (AbstractLabel):
    __json_type_name__ = 'point'

    def __init__(self, position_xy, object_id=None, classification=None):
        """
        Constructor

        :param position_xy: position of point as a (2,) NumPy array providing the x and y co-ordinates
        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        super(PointLabel, self).__init__(object_id, classification)
        self.position_xy = np.array(position_xy).astype(float)

    @property
    def dependencies(self):
        return []

    def bounding_box(self, ctx=None):
        point_radius = ctx.point_radius if ctx is not None else 0.0
        return self.position_xy - point_radius, self.position_xy + point_radius

    def _warp(self, xform_fn, object_table):
        warped_pos = xform_fn(self.position_xy[None, :])
        return PointLabel(warped_pos[0, :], self.object_id, self.classification)

    def _render_mask(self, img, fill, dx=0.0, dy=0.0, ctx=None):
        point_radius = ctx.point_radius if ctx is not None else 0.0

        x = self.position_xy[0] + dx
        y = self.position_xy[1] + dy

        if point_radius == 0.0:
            ImageDraw.Draw(img).point((x, y), fill=1)
        else:
            ellipse = [(x-point_radius, y-point_radius),
                       (x+point_radius, y+point_radius)]
            if fill:
                ImageDraw.Draw(img).ellipse(ellipse, outline=1, fill=1)
            else:
                ImageDraw.Draw(img).ellipse(ellipse, outline=1, fill=0)

    def to_json(self):
        js = super(PointLabel, self).to_json()
        js['position'] = dict(x=self.position_xy[0], y=self.position_xy[1])
        return js

    def __str__(self):
        return 'PointLabel(object_id={}, classification={}, position_xy={})'.format(
            self.object_id, self.classification, self.position_xy.tolist()
        )

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        pos_xy = np.array([label_json['position']['x'], label_json['position']['y']])
        return PointLabel(pos_xy, label_json.get('object_id'), label_json['label_class'])


@label_cls
class PolygonLabel (AbstractLabel):
    __json_type_name__ = 'polygon'

    def __init__(self, vertices, object_id=None, classification=None):
        """
        Constructor

        :param vertices: vertices as a (N,2) NumPy array providing the [x, y] co-ordinates
        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        super(PolygonLabel, self).__init__(object_id, classification)
        vertices = np.array(vertices).astype(float)
        self.vertices = vertices

    @property
    def dependencies(self):
        return []

    def bounding_box(self, ctx=None):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def _warp(self, xform_fn, object_table):
        warped_verts = xform_fn(self.vertices)
        return PolygonLabel(warped_verts, self.object_id, self.classification)

    def _render_mask(self, img, fill, dx=0.0, dy=0.0, ctx=None):
        # Rendering helper function: create a binary mask for a given label

        # Polygonal label
        if len(self.vertices) >= 3:
            vertices = self.vertices + np.array([[dx, dy]])
            polygon = [tuple(v) for v in vertices]

            if fill:
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            else:
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)

    def to_json(self):
        js = super(PolygonLabel, self).to_json()
        js['vertices'] = [dict(x=self.vertices[i,0], y=self.vertices[i,1]) for i in range(len(self.vertices))]
        return js

    def __str__(self):
        return 'PolygonLabel(object_id={}, classification={}, vertices={})'.format(
            self.object_id, self.classification, self.vertices
        )

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        verts = np.array([[v['x'], v['y']] for v in label_json['vertices']])
        return PolygonLabel(verts, label_json.get('object_id'), label_json['label_class'])


@label_cls
class BoxLabel (AbstractLabel):
    __json_type_name__ = 'box'

    def __init__(self, centre_xy, size_xy, object_id=None, classification=None):
        """
        Constructor

        :param centre_xy: centre of box as a (2,) NumPy array providing the x and y co-ordinates
        :param size_xy: size of box as a (2,) NumPy array providing the x and y co-ordinates
        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        super(BoxLabel, self).__init__(object_id, classification)
        self.centre_xy = np.array(centre_xy).astype(float)
        self.size_xy = np.array(size_xy).astype(float)

    @property
    def dependencies(self):
        return []

    def bounding_box(self, ctx=None):
        return self.centre_xy - self.size_xy, self.centre_xy + self.size_xy

    def _warp(self, xform_fn, object_table):
        corners = np.array([
            self.centre_xy + self.size_xy * -1,
            self.centre_xy + self.size_xy * np.array([1, -1]),
            self.centre_xy + self.size_xy,
            self.centre_xy + self.size_xy * np.array([-1, 1]),
        ])
        xf_corners = xform_fn(corners)
        lower = xf_corners.min(axis=0)
        upper = xf_corners.max(axis=0)
        xf_centre = (lower + upper) * 0.5
        xf_size = upper - lower
        return BoxLabel(xf_centre, xf_size, self.object_id, self.classification)

    def _render_mask(self, img, fill, dx=0.0, dy=0.0, ctx=None):
        # Rendering helper function: create a binary mask for a given label

        centre = self.centre_xy + np.array([dx, dy])
        lower = centre - self.size_xy * 0.5
        upper = centre + self.size_xy * 0.5

        if fill:
            ImageDraw.Draw(img).rectangle([lower, upper], outline=1, fill=1)
        else:
            ImageDraw.Draw(img).rectangle([lower, upper], outline=1, fill=0)

    def to_json(self):
        js = super(BoxLabel, self).to_json()
        js['centre'] = dict(x=self.centre_xy[0], y=self.centre_xy[1])
        js['size'] = dict(x=self.size_xy[0], y=self.size_xy[1])
        return js

    def __str__(self):
        return 'BoxLabel(object_id={}, classification={}, centre_xy={}, size_xy={})'.format(
            self.object_id, self.classification, self.centre_xy.tolist(), self.size_xy.tolist()
        )

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        centre = np.array([label_json['centre']['x'], label_json['centre']['y']])
        size = np.array([label_json['size']['x'], label_json['size']['y']])
        return BoxLabel(centre, size, label_json.get('object_id'), label_json['label_class'])


@label_cls
class CompositeLabel (AbstractLabel):
    __json_type_name__ = 'composite'

    def __init__(self, components, object_id=None, classification=None):
        """
        Constructor

        :param components: a list of label objects that are members of the composite label
        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        super(CompositeLabel, self).__init__(object_id, classification)
        self.components = components

    @property
    def dependencies(self):
        return self.components

    def bounding_box(self, ctx=None):
        return None, None

    def _warp(self, xform_fn, object_table):
        warped_components = []
        for comp in self.components:
            if comp.object_id in object_table:
                warped_comp = object_table[comp.object_id]
            else:
                warped_comp = comp.warped(xform_fn, object_table)
            warped_components.append(warped_comp)
        return CompositeLabel(warped_components, self.object_id, self.classification)

    def render_mask(self, width, height, fill, dx=0.0, dy=0.0, ctx=None):
        return None

    def to_json(self):
        js = super(CompositeLabel, self).to_json()
        js['components'] = [component.object_id for component in self.components]
        return js

    def __str__(self):
        return 'CompositeLabel(object_id={}, classification={}, ids(components)={}'.format(
            self.object_id, self.classification, [c.object_id for c in self.components]
        )

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        components = [object_table.get(obj_id) for obj_id in label_json['components']]
        components = [comp for comp in components if comp is not None]
        return CompositeLabel(components, label_json.get('object_id'), label_json['label_class'])


@label_cls
class GroupLabel (AbstractLabel):
    __json_type_name__ = 'group'

    def __init__(self, component_labels, object_id=None, classification=None):
        """
        Constructor

        :param component_labels: a list of label objects that are members of the group label
        :param object_id: a unique integer object ID or None
        :param classification: a str giving the label's ground truth classification
        """
        super(GroupLabel, self).__init__(object_id, classification)
        self.component_labels = component_labels

    def flatten(self):
        for comp in self.component_labels:
            for f in comp.flatten():
                yield f

    def bounding_box(self, ctx=None):
        lowers, uppers = list(zip(*[comp.bounding_box(ctx) for comp in self.component_labels]))
        lowers = [x for x in lowers if x is not None]
        uppers = [x for x in uppers if x is not None]
        if len(lowers) > 0 and len(uppers) > 0:
            return np.array(lowers).min(axis=0), np.array(uppers).max(axis=0)
        else:
            return None, None

    def _warp(self, xform_fn, object_table):
        comps = [comp.warped(xform_fn, object_table) for comp in self.component_labels]
        return GroupLabel(comps, self.object_id, self.classification)

    def _render_mask(self, img, fill, dx=0.0, dy=0.0, ctx=None):
        for label in self.component_labels:
            label._render_mask(img, fill, dx, dy, ctx)

    def to_json(self):
        js = super(GroupLabel, self).to_json()
        js['component_models'] = [component.to_json() for component in self.component_labels]
        return js

    def __str__(self):
        return 'GroupLabel(object_id={}, classification={}, component_labels={}'.format(
            self.object_id, self.classification, self.component_labels
        )

    @classmethod
    def new_instance_from_json(cls, label_json, object_table):
        components = [AbstractLabel.from_json(comp, object_table)
                      for comp in label_json['component_models']]
        return CompositeLabel(components, label_json.get('object_id'), label_json['label_class'])



class ObjectTable (object):
    def __init__(self, objects=None):
        self._object_id_to_obj = {}
        self._next_object_id = 1

        if objects is not None:
            # Register objects with object IDs
            for obj in objects:
                self.register(obj)

            # Allocate object IDs to objects with no ID
            self._alloc_object_ids(objects)

    def _alloc_object_ids(self, objects):
        for obj in objects:
            if obj.object_id is None:
                self._alloc_id(obj)

    def _alloc_id(self, obj):
        obj_id = self._next_object_id
        self._next_object_id += 1
        obj.object_id = obj_id
        self._object_id_to_obj[obj_id] = obj
        return obj_id

    def register(self, obj):
        obj_id = obj.object_id
        if obj_id is not None:
            if obj_id in self._object_id_to_obj:
                raise ValueError('Duplicate object ID')
            self._object_id_to_obj[obj_id] = obj
            self._next_object_id = max(self._next_object_id, obj_id + 1)

    def __getitem__(self, obj_id):
        if obj_id is None:
            return None
        else:
            return self._object_id_to_obj[obj_id]

    def get(self, obj_id, default=None):
        if obj_id is None:
            return None
        else:
            return self._object_id_to_obj.get(obj_id, default)

    def __contains__(self, obj_id):
        return obj_id in self._object_id_to_obj


class ImageLabels (object):
    """
    Represents labels in vector format, stored in JSON form. Has methods for
    manipulating and rendering them.

    """
    def __init__(self, labels, obj_table=None):
        self.labels = labels
        if obj_table is None:
            obj_table = ObjectTable(list(self.flatten()))
        self._obj_table = obj_table


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item]


    def flatten(self):
        for lab in self.labels:
            for f in lab.flatten():
                yield f


    def warp(self, xform_fn):
        """
        Warp the labels given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an `ImageLabels` instance that contains the warped labels
        """
        warped_obj_table = ObjectTable()
        warped_labels = [lab.warped(xform_fn, warped_obj_table) for lab in self.labels]
        return  ImageLabels(warped_labels, obj_table=warped_obj_table)


    def render_labels(self, label_classes, image_shape, pixels_as_vectors=False, fill=True, ctx=None):
        """
        Render the labels to create a label image

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
        :param image_shape: `(height, width)` tuple specifying the shape of the image to be returned
        :param pixels_as_vectors: If `False`, return an (height,width) array of dtype=int with pixels numbered
            according to their label. If `True`, return a (height,width,n_labels) array of dtype=float32 with each pixel
            being a feature vector that gives the weight of each label, where n_labels is `len(label_classes)`
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :return: (H,W) array with dtype=int if pixels_as_vectors is False, otherwise (H,W,n_labels) with dtype=float32
        """
        if isinstance(label_classes, list) or isinstance(label_classes, tuple):
            cls_to_index = {}
            for i, cls in enumerate(label_classes):
                if isinstance(cls, LabelClass):
                    cls_to_index[cls.name] = i
                elif isinstance(cls, six.string_types)  or  cls is None:
                    cls_to_index[cls] = i
                elif isinstance(cls, list)  or  isinstance(cls, tuple):
                    for c in cls:
                        if isinstance(c, LabelClass):
                            cls_to_index[c.name] = i
                        elif isinstance(c, six.string_types)  or  c is None:
                            cls_to_index[c] = i
                        else:
                            raise TypeError('Item {0} in label_classes is a list that contains an item that is not a '
                                            'LabelClass or a string but a {1}'.format(i, type(c).__name__))
                else:
                    raise TypeError('Item {0} in label_classes is not a LabelClass, string or list, '
                                    'but a {1}'.format(i, type(cls).__name__))
        else:
            raise TypeError('label_classes must be a sequence that can contain LabelClass instances, strings or '
                            'sub-sequences of the former')


        height, width = image_shape

        if pixels_as_vectors:
            label_image = np.zeros((height, width, len(label_classes)), dtype='float32')
        else:
            label_image = np.zeros((height, width), dtype=int)

        for label in self.flatten():
            label_cls_n = cls_to_index.get(label.classification, None)
            if label_cls_n is not None:
                mask = label.render_mask(width, height, fill, ctx=ctx)
                if mask is not None:
                    if pixels_as_vectors:
                        label_image[:,:,label_cls_n] += mask
                        label_image[:,:,label_cls_n] = np.clip(label_image[:,:,label_cls_n], 0.0, 1.0)
                    else:
                        label_image[mask >= 0.5] = label_cls_n + 1

        return label_image


    def render_individual_labels(self, label_classes, image_shape, fill=True, ctx=None):
        """
        Render individual labels to create a label image.
        The resulting image is a multi-channel image, with a channel for each class in `label_classes`.
        Each individual label's class is used to select the channel that it is rendered into.
        Each label is given a different index that is rendered into the resulting image.

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
            Each entry within label_classes will have a corresponding channel in the output image
        :param image_shape: `(height, width)` tuple specifying the shape of the image to be returned
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :param image_shape: `None`, or a `(height, width)` tuple specifying the shape of the image to be rendered
        :return: tuple of (label_image, label_counts) where:
            label_image is a (H,W,C) array with dtype=int
            label_counts is a 1D array of length C (number of channels) that contains the number of labels drawn for each channel; effectively the maximum value found in each channel
        """
        # Create `cls_to_channel`
        if isinstance(label_classes, list) or isinstance(label_classes, tuple):
            cls_to_channel = {}
            for i, cls in enumerate(label_classes):
                if isinstance(cls, LabelClass):
                    cls_to_channel[cls.name] = i
                elif isinstance(cls, six.string_types)  or  cls is None:
                    cls_to_channel[cls] = i
                elif isinstance(cls, list)  or  isinstance(cls, tuple):
                    for c in cls:
                        if isinstance(c, LabelClass):
                            cls_to_channel[c.name] = i
                        elif isinstance(c, six.string_types):
                            cls_to_channel[c] = i
                        else:
                            raise TypeError('Item {0} in label_classes is a list that contains an item that is not a '
                                            'LabelClass or a string but a {1}'.format(i, type(c).__name__))
                else:
                    raise TypeError('Item {0} in label_classes is not a LabelClass, string or list, '
                                    'but a {1}'.format(i, type(cls).__name__))
        else:
            raise TypeError('label_classes must be a sequence that can contain LabelClass instances, strings or '
                            'sub-sequences of the former')


        height, width = image_shape

        label_image = np.zeros((height, width, len(label_classes)), dtype=int)

        channel_label_count = [0] * len(label_classes)

        for label in self.flatten():
            label_channel = cls_to_channel.get(label.classification, None)
            if label_channel is not None:
                mask = label.render_mask(width, height, fill, ctx=ctx)
                if mask is not None:
                    value = channel_label_count[label_channel]
                    channel_label_count[label_channel] += 1

                    label_image[mask >= 0.5, label_channel] = value + 1

        return label_image, np.array(channel_label_count)


    def extract_label_images(self, image_2d, label_class_set=None, ctx=None):
        """
        Extract an image of each labelled entity from a given image.
        The resulting image is the original image masked with an alpha channel that results from rendering the label

        :param image_2d: the image from which to extract images of labelled objects
        :param label_class_set: a sequence of classes whose labels should be rendered, or None for all labels
        :return: a list of (H,W,C) image arrays
        """
        image_shape = image_2d.shape[:2]

        label_images = []

        for label in self.flatten():
            if label_class_set is None  or  label.classification in label_class_set:
                bounds = label.bounding_box(ctx=ctx)

                if bounds[0] is not None and bounds[1] is not None:
                    lx = int(math.floor(bounds[0][0]))
                    ly = int(math.floor(bounds[0][1]))
                    ux = int(math.ceil(bounds[1][0]))
                    uy = int(math.ceil(bounds[1][1]))

                    # Given that the images and labels may have been warped by a transformation,
                    # there is no guarantee that they lie within the bounds of the image
                    lx = max(min(lx, image_shape[1]), 0)
                    ux = max(min(ux, image_shape[1]), 0)
                    ly = max(min(ly, image_shape[0]), 0)
                    uy = max(min(uy, image_shape[0]), 0)

                    w = ux - lx
                    h = uy - ly

                    if w > 0 and h > 0:

                        mask = label.render_mask(w, h, fill=True, dx=float(-lx), dy=float(-ly), ctx=ctx)
                        if mask is not None and (mask > 0).any():
                            img_box = image_2d[ly:uy, lx:ux]
                            if len(img_box.shape) == 2:
                                # Convert greyscale image to RGB:
                                img_box = gray2rgb(img_box)
                            # Append the mask as an alpha channel
                            object_img = np.append(img_box, mask[:,:,None], axis=2)

                            label_images.append(object_img)

        return label_images


    def to_json(self):
        return [lab.to_json() for lab in self.labels]

    @staticmethod
    def from_json(label_data_js):
        """
        Labels in JSON format

        :param label_data_js: either a list of labels in JSON format or a dict that maps the key `'labels'` to a list
        of labels in JSON form. The dict format will match the format stored in JSON label files.

        :return: an `ImageLabels` instance
        """
        if isinstance(label_data_js, dict):
            if 'labels' not in label_data_js:
                raise ValueError('label_js should be a list or a dict containing a \'labels\' key')
            labels = label_data_js['labels']
            if not isinstance(labels, list):
                raise TypeError('labels[\'labels\'] should be a list')
        elif isinstance(label_data_js, list):
            labels = label_data_js
        else:
            raise ValueError('label_js should be a list or a dict containing a \'labels\' key')

        obj_table = ObjectTable()
        labs = [AbstractLabel.from_json(label, obj_table) for label in labels]
        return ImageLabels(labs, obj_table=obj_table)


    @staticmethod
    def from_file(f):
        if isinstance(f, six.string_types):
            f = open(f, 'r')
        elif isinstance(f, io.IOBase):
            pass
        else:
            raise TypeError('f should be a path as a string or a file')
        return ImageLabels.from_json(json.load(f))



    @classmethod
    def from_contours(cls, list_of_contours, label_classes=None):
        """
        Convert a list of contours to an `ImageLabels` instance.

        :param list_of_contours: list of contours, where each contour is an `(N,2)` numpy array.
                where `N` is the number of vertices, each of which is a `(y,x)` pair.
        :param label_classes: [optional] a list of the same length as `list_of_contours` that provides
                the label class of each contour
        :return: an `ImageLabels` instance containing the labels extracted from the contours
        """
        obj_table = ObjectTable()
        labels = []
        if not isinstance(label_classes, list):
            label_classes = [label_classes] * len(list_of_contours)
        for contour, lcls in zip(list_of_contours, label_classes):
            vertices = np.array([[contour[i][1], contour[i][0]] for i in range(len(contour))])
            label = PolygonLabel(vertices, classification=lcls)
            obj_table.register(label)
            labels.append(label)
        return cls(labels, obj_table=obj_table)


    @classmethod
    def from_label_image(cls, labels):
        """
        Convert a integer label mask image to an `ImageLabels` instance.

        :param labels: a `(h,w)` numpy array of dtype `int32` that gives an integer label for each
                pixel in the image. Label values start at 1; pixels with a value of 0 will not be
                included in the returned labels.
        :return: an `ImageLabels` instance containing the labels extracted from the label mask image
        """
        contours = []
        for i in range(1, labels.max()+1):
            lmask = labels == i

            if lmask.sum() > 0:
                mask_positions = np.argwhere(lmask)
                (ystart, xstart), (ystop, xstop) = mask_positions.min(0), mask_positions.max(0) + 1

                if ystop >= ystart+1 and xstop >= xstart+1:
                    mask_trim = lmask[ystart:ystop, xstart:xstop]
                    mask_trim = pad(mask_trim, [(1,1), (1,1)], mode='constant').astype(np.float32)
                    cs = find_contours(mask_trim, 0.5)
                    for contour in cs:
                        simp = _simplify_contour(contour + np.array((ystart, xstart)) - np.array([[1.0, 1.0]]))
                        if simp is not None:
                            contours.append(simp)
        return cls.from_contours(contours)



class AbsractLabelledImage (object):
    def __init__(self):
        pass


    @property
    def pixels(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    @property
    def image_size(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def data_and_mime_type_and_size(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))


    @property
    def labels(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    @labels.setter
    def labels(self, l):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def has_labels(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    @property
    def labels_json(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    @labels_json.setter
    def labels_json(self, l):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))


    @property
    def complete(self):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    @complete.setter
    def complete(self, c):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))


    def get_label_data_for_tool(self):
        return self.labels_json, self.complete

    def set_label_data_from_tool(self, labels_js, complete):
        self.complete = complete
        self.labels_json = labels_js



    def warped(self, projection, sz_px):
        warped_pixels = transform.warp(self.pixels, projection.inverse)[:int(sz_px[0]),:int(sz_px[1])].astype('float32')
        warped_labels = self.labels._warp(projection)
        return InMemoryLabelledImage(warped_pixels, warped_labels)


    def render_labels(self, label_classes, pixels_as_vectors=False, fill=True):
        """
        Render the labels to create a label image

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
        :param pixels_as_vectors: If `False`, return an (height,width) array of dtype=int with pixels numbered
            according to their label. If `True`, return a (height,width,n_labels) array of dtype=float32 with each pixel
            being a feature vector that gives the weight of each label, where n_labels is `len(label_classes)`
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :return: (H,W) array with dtype=int if pixels_as_vectors is False, otherwise (H,W,n_labels) with dtype=float32
        """
        return self.labels.render_labels(label_classes, self.image_size,
                                         pixels_as_vectors=pixels_as_vectors, fill=fill)


    def render_individual_labels(self, label_classes, fill=True):
        """
        Render individual labels to create a label image.
        The resulting image is a multi-channel image, with a channel for each class in `label_classes`.
        Each individual label's class is used to select the channel that it is rendered into.
        Each label is given a different index that is rendered into the resulting image.

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
            Each entry within label_classes will have a corresponding channel in the output image
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :param image_shape: `None`, or a `(height, width)` tuple specifying the shape of the image to be rendered
        :return: tuple of (label_image, label_counts) where:
            label_image is a (H,W,C) array with dtype=int
            label_counts is a 1D array of length C (number of channels) that contains the number of labels drawn for each channel; effectively the maximum value found in each channel
        """
        return self.labels.render_individual_labels(label_classes, self.image_size, fill=fill)


    def extract_label_images(self, label_class_set=None):
        """
        Extract an image of each labelled entity.
        The resulting image is the original image masked with an alpha channel that results from rendering the label

        :param label_class_set: a sequence of classes whose labels should be rendered, or None for all labels
        :return: a list of (H,W,C) image arrays
        """
        return self.labels.extract_label_images(self.pixels, label_class_set=label_class_set)



class InMemoryLabelledImage (AbsractLabelledImage):
    def __init__(self, pixels, labels=None, complete=False):
        super(InMemoryLabelledImage, self).__init__()
        if labels is None:
            labels = ImageLabels([])
        self.__pixels = pixels
        self.__labels = labels
        self.__complete = complete


    @property
    def pixels(self):
        return self.__pixels

    def image_size(self):
        return self.__pixels.shape[:2]

    def data_and_mime_type_and_size(self):
        buf = io.BytesIO()
        imsave(buf, self.__pixels, format='png')
        return buf.getvalue(), 'image/png', int(self.__pixels.shape[1]), int(self.__pixels.shape[0])



    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, l):
        self.__labels = l

    def has_labels(self):
        return True

    @property
    def labels_json(self):
        return self.__labels.to_json()

    @labels_json.setter
    def labels_json(self, l):
        self.__labels = ImageLabels.from_json(l)


    @property
    def complete(self):
        return self.__complete

    @complete.setter
    def complete(self, c):
        self.__complete = c


class PersistentLabelledImage (AbsractLabelledImage):
    def __init__(self, image_path, labels_path, readonly=False):
        super(PersistentLabelledImage, self).__init__()
        self.__image_path = image_path
        self.__labels_path = labels_path
        self.__pixels = None

        self.__labels_json = None
        self.__complete = None
        self.__readonly = readonly



    @property
    def pixels(self):
        if self.__pixels is None:
            self.__pixels = img_as_float(imread(self.__image_path))
        return self.__pixels

    @property
    def image_size(self):
        if self.__pixels is not None:
            return self.__pixels.shape[:2]
        else:
            i = Image.open(self.__image_path)
            return i.size[1], i.size[0]

    def data_and_mime_type_and_size(self):
        if os.path.exists(self.__image_path):
            with open(self.__image_path, 'rb') as img:
                shape = self.image_size
                return img.read(), mimetypes.guess_type(self.__image_path)[0], int(shape[1]), int(shape[0])


    @property
    def image_path(self):
        return self.__image_path

    @property
    def image_filename(self):
        return os.path.basename(self.__image_path)

    @property
    def image_name(self):
        return os.path.splitext(self.image_filename)[0]



    @property
    def labels(self):
        return ImageLabels.from_json(self.labels_json)

    @labels.setter
    def labels(self, l):
        self.labels_json = l.to_json()


    @property
    def labels_json(self):
        labels_js, complete = self._get_labels()
        return labels_js

    @labels_json.setter
    def labels_json(self, labels_json):
        self._set_labels(labels_json, self.__complete)


    @property
    def complete(self):
        labels_js, complete = self._get_labels()
        return complete

    @complete.setter
    def complete(self, c):
        self._set_labels(self.__labels_json, c)


    def has_labels(self):
        return os.path.exists(self.__labels_path)


    def get_label_data_for_tool(self):
        return self._get_labels()

    def set_label_data_from_tool(self, labels_js, complete):
        self._set_labels(labels_js, complete)



    def _get_labels(self):
        if self.__labels_json is None:
            if os.path.exists(self.__labels_path):
                with open(self.__labels_path, 'r') as f:
                    try:
                        js = json.load(f)
                        self.__labels_json, self.__complete = self._unwrap_labels(js)
                    except ValueError:
                        traceback.print_exc()
                        pass
        return self.__labels_json, self.__complete


    def _set_labels(self, labels_js, complete):
        if not self.__readonly:
            if labels_js is None  or  (len(labels_js) == 0 and not complete):
                # No data; delete the file
                if os.path.exists(self.__labels_path):
                    os.remove(self.__labels_path)
            else:
                with open(self.__labels_path, 'w') as f:
                    wrapped = self.__wrap_labels(os.path.split(self.image_path)[1], labels_js, complete)
                    json.dump(wrapped, f, indent=3)
        self.__labels_json = labels_js
        self.__complete = complete




    @staticmethod
    def __wrap_labels(image_path, labels, complete):
        image_filename = os.path.split(image_path)[1]
        return {'image_filename': image_filename,
                'complete': complete,
                'labels': labels}

    @staticmethod
    def _unwrap_labels(wrapped_labels):
        if isinstance(wrapped_labels, dict):
            return wrapped_labels['labels'], wrapped_labels.get('complete', False)
        elif isinstance(wrapped_labels, list):
            return wrapped_labels, False
        else:
            raise TypeError('Labels loaded from file must either be a dict or a list, '
                            'not a {0}'.format(type(wrapped_labels)))


    @staticmethod
    def __compute_labels_path(path, labels_dir=None):
        p = os.path.splitext(path)[0] + '__labels.json'
        if labels_dir is not None:
            p = os.path.join(labels_dir, os.path.split(p)[1])
        return p


    @classmethod
    def for_directory(cls, dir_path, image_filename_pattern='*.png', with_labels_only=False, labels_dir=None, readonly=False):
        image_paths = glob.glob(os.path.join(dir_path, image_filename_pattern))
        limgs = []
        for img_path in image_paths:
            labels_path = cls.__compute_labels_path(img_path, labels_dir=labels_dir)
            if not with_labels_only or os.path.exists(labels_path):
                limgs.append(PersistentLabelledImage(img_path, labels_path, readonly=readonly))
        return limgs


class LabelledImageFile (AbsractLabelledImage):
    def __init__(self, path, labels=None, complete=False, on_set_labels=None):
        super(LabelledImageFile, self).__init__()
        if labels is None:
            labels = ImageLabels([])
        self.__labels = labels
        self.__complete = complete
        self.__image_path = path
        self.__pixels = None
        self.__on_set_labels = on_set_labels



    @property
    def pixels(self):
        if self.__pixels is None:
            self.__pixels = img_as_float(imread(self.__image_path))
        return self.__pixels

    @property
    def image_size(self):
        if self.__pixels is not None:
            return self.__pixels.shape[:2]
        else:
            i = Image.open(self.__image_path)
            return i.size[1], i.size[0]

    def data_and_mime_type_and_size(self):
        if os.path.exists(self.__image_path):
            with open(self.__image_path, 'rb') as img:
                shape = self.image_size
                return img.read(), mimetypes.guess_type(self.__image_path)[0], int(shape[1]), int(shape[0])


    @property
    def image_path(self):
        return self.__image_path

    @property
    def image_filename(self):
        return os.path.basename(self.__image_path)

    @property
    def image_name(self):
        return os.path.splitext(self.image_filename)[0]



    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, l):
        self.__labels = l
        if self.__on_set_labels is not None:
            self.__on_set_labels(self.__labels)


    def has_labels(self):
        return True


    @property
    def labels_json(self):
        return self.__labels.to_json()

    @labels_json.setter
    def labels_json(self, l):
        self.__labels = ImageLabels.from_json(l)
        if self.__on_set_labels is not None:
            self.__on_set_labels(self.__labels)


    @property
    def complete(self):
        return self.__complete

    @complete.setter
    def complete(self, c):
        self.__complete = c



def shuffle_images_without_labels(labelled_images):
    with_labels = [img   for img in labelled_images   if img.has_labels()]
    without_labels = [img   for img in labelled_images   if not img.has_labels()]
    random.shuffle(without_labels)
    return with_labels + without_labels

