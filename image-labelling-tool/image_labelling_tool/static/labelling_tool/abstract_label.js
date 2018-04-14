/*
The MIT License (MIT)

Copyright (c) 2015 University of East Anglia, Norwich, UK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Developed by Geoffrey French in collaboration with Dr. M. Fisher and
Dr. M. Mackiewicz.
 */
/// <reference path="../d3.d.ts" />
/// <reference path="./math_primitives.ts" />
/// <reference path="./root_label_view.ts" />
var labelling_tool;
(function (labelling_tool) {
    /*
    Label visibility
     */
    var LabelVisibility;
    (function (LabelVisibility) {
        LabelVisibility[LabelVisibility["HIDDEN"] = 0] = "HIDDEN";
        LabelVisibility[LabelVisibility["FAINT"] = 1] = "FAINT";
        LabelVisibility[LabelVisibility["FULL"] = 2] = "FULL";
    })(LabelVisibility = labelling_tool.LabelVisibility || (labelling_tool.LabelVisibility = {}));
    /*
    Abstract label entity
     */
    var AbstractLabelEntity = (function () {
        function AbstractLabelEntity(view, model) {
            this.root_view = view;
            this.model = model;
            this._attached = this._hover = this._selected = false;
            this._event_listeners = [];
            this.parent_entity = null;
            this.entity_id = AbstractLabelEntity.entity_id_counter++;
        }
        AbstractLabelEntity.prototype.add_event_listener = function (listener) {
            this._event_listeners.push(listener);
        };
        AbstractLabelEntity.prototype.remove_event_listener = function (listener) {
            var i = this._event_listeners.indexOf(listener);
            if (i !== -1) {
                this._event_listeners.splice(i, 1);
            }
        };
        AbstractLabelEntity.prototype.set_parent = function (parent) {
            this.parent_entity = parent;
        };
        AbstractLabelEntity.prototype.get_entity_id = function () {
            return this.entity_id;
        };
        AbstractLabelEntity.prototype.attach = function () {
            this.root_view._register_entity(this);
            this._attached = true;
        };
        AbstractLabelEntity.prototype.detach = function () {
            this._attached = false;
            this.root_view._unregister_entity(this);
        };
        AbstractLabelEntity.prototype.destroy = function () {
            if (this.parent_entity !== null) {
                this.parent_entity.remove_child(this);
            }
            this.root_view.shutdown_entity(this);
        };
        AbstractLabelEntity.prototype.update = function () {
        };
        AbstractLabelEntity.prototype.commit = function () {
        };
        AbstractLabelEntity.prototype.hover = function (state) {
            this._hover = state;
            this._update_style();
        };
        AbstractLabelEntity.prototype.select = function (state) {
            this._selected = state;
            this._update_style();
        };
        AbstractLabelEntity.prototype.notify_hide_labels_change = function () {
            this._update_style();
        };
        AbstractLabelEntity.prototype.get_label_type_name = function () {
            return this.model.label_type;
        };
        AbstractLabelEntity.prototype.get_label_class = function () {
            return this.model.label_class;
        };
        AbstractLabelEntity.prototype.set_label_class = function (label_class) {
            this.model.label_class = label_class;
            this._update_style();
            this.commit();
        };
        AbstractLabelEntity.prototype._update_style = function () {
        };
        ;
        AbstractLabelEntity.prototype._outline_colour = function () {
            if (this._selected) {
                if (this._hover) {
                    return new labelling_tool.Colour4(255, 0, 128, 1.0);
                }
                else {
                    return new labelling_tool.Colour4(255, 0, 0, 1.0);
                }
            }
            else {
                if (this._hover) {
                    return new labelling_tool.Colour4(0, 255, 128, 1.0);
                }
                else {
                    return new labelling_tool.Colour4(255, 255, 0, 1.0);
                }
            }
        };
        AbstractLabelEntity.prototype.compute_centroid = function () {
            return null;
        };
        AbstractLabelEntity.prototype.compute_bounding_box = function () {
            return null;
        };
        ;
        AbstractLabelEntity.prototype.contains_pointer_position = function (point) {
            return false;
        };
        AbstractLabelEntity.prototype.distance_to_point = function (point) {
            return null;
        };
        ;
        AbstractLabelEntity.prototype.notify_model_destroyed = function (model_id) {
        };
        ;
        AbstractLabelEntity.entity_id_counter = 0;
        return AbstractLabelEntity;
    }());
    labelling_tool.AbstractLabelEntity = AbstractLabelEntity;
    /*
    Map label type to entity constructor
     */
    var label_type_to_entity_factory = {};
    /*
    Register label entity factory
     */
    function register_entity_factory(label_type_name, factory) {
        label_type_to_entity_factory[label_type_name] = factory;
    }
    labelling_tool.register_entity_factory = register_entity_factory;
    /*
    Construct entity for given label model.
    Uses the map above to choose the appropriate constructor
     */
    function new_entity_for_model(root_view, label_model) {
        var factory = label_type_to_entity_factory[label_model.label_type];
        return factory(root_view, label_model);
    }
    labelling_tool.new_entity_for_model = new_entity_for_model;
})(labelling_tool || (labelling_tool = {}));
