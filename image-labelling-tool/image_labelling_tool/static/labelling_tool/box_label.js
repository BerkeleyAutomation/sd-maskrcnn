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
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
/// <reference path="./math_primitives.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./abstract_tool.ts" />
/// <reference path="./select_tools.ts" />
var labelling_tool;
(function (labelling_tool) {
    function new_BoxLabelModel(centre, size) {
        return { label_type: 'box', label_class: null, centre: centre, size: size };
    }
    function BoxLabel_box(label) {
        var lower = { x: label.centre.x - label.size.x * 0.5, y: label.centre.y - label.size.y * 0.5 };
        var upper = { x: label.centre.x + label.size.x * 0.5, y: label.centre.y + label.size.y * 0.5 };
        return new labelling_tool.AABox(lower, upper);
    }
    /*
    Box label entity
     */
    var BoxLabelEntity = (function (_super) {
        __extends(BoxLabelEntity, _super);
        function BoxLabelEntity(view, model) {
            return _super.call(this, view, model) || this;
        }
        BoxLabelEntity.prototype.attach = function () {
            _super.prototype.attach.call(this);
            this._rect = this.root_view.world.append("rect")
                .attr("x", 0).attr("y", 0)
                .attr("width", 0).attr("height", 0);
            this.update();
            var self = this;
            this._rect.on("mouseover", function () {
                self._on_mouse_over_event();
            }).on("mouseout", function () {
                self._on_mouse_out_event();
            });
            this._update_style();
        };
        ;
        BoxLabelEntity.prototype.detach = function () {
            this._rect.remove();
            this._rect = null;
            _super.prototype.detach.call(this);
        };
        BoxLabelEntity.prototype._on_mouse_over_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_in(this);
            }
        };
        BoxLabelEntity.prototype._on_mouse_out_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_out(this);
            }
        };
        BoxLabelEntity.prototype.update = function () {
            var box = BoxLabel_box(this.model);
            var size = box.size();
            this._rect
                .attr('x', box.lower.x).attr('y', box.lower.y)
                .attr('width', size.x).attr('height', size.y);
        };
        BoxLabelEntity.prototype.commit = function () {
            this.root_view.commit_model(this.model);
        };
        BoxLabelEntity.prototype._update_style = function () {
            if (this._attached) {
                var stroke_colour = this._outline_colour();
                if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.HIDDEN) {
                    this._rect.attr("visibility", "hidden");
                }
                else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FAINT) {
                    stroke_colour = stroke_colour.with_alpha(0.2);
                    this._rect.attr("style", "fill:none;stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this._rect.attr("visibility", "visible");
                }
                else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FULL) {
                    var circle_fill_colour = this.root_view.view.colour_for_label_class(this.model.label_class);
                    if (this._hover) {
                        circle_fill_colour = circle_fill_colour.lighten(0.4);
                    }
                    circle_fill_colour = circle_fill_colour.with_alpha(0.35);
                    stroke_colour = stroke_colour.with_alpha(0.5);
                    this._rect.attr("style", "fill:" + circle_fill_colour.to_rgba_string() + ";stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this._rect.attr("visibility", "visible");
                }
            }
        };
        BoxLabelEntity.prototype.compute_centroid = function () {
            return this.model.centre;
        };
        ;
        BoxLabelEntity.prototype.compute_bounding_box = function () {
            return BoxLabel_box(this.model);
        };
        ;
        BoxLabelEntity.prototype.contains_pointer_position = function (point) {
            return this.compute_bounding_box().contains_point(point);
        };
        BoxLabelEntity.prototype.distance_to_point = function (point) {
            return BoxLabel_box(this.model).distance_to(point);
        };
        return BoxLabelEntity;
    }(labelling_tool.AbstractLabelEntity));
    labelling_tool.BoxLabelEntity = BoxLabelEntity;
    labelling_tool.register_entity_factory('box', function (root_view, model) {
        return new BoxLabelEntity(root_view, model);
    });
    /*
    Draw box tool
     */
    var DrawBoxTool = (function (_super) {
        __extends(DrawBoxTool, _super);
        function DrawBoxTool(view, entity) {
            var _this = _super.call(this, view) || this;
            _this.entity = entity;
            _this._start_point = null;
            _this._current_point = null;
            return _this;
        }
        DrawBoxTool.prototype.on_init = function () {
        };
        ;
        DrawBoxTool.prototype.on_shutdown = function () {
        };
        ;
        DrawBoxTool.prototype.on_switch_in = function (pos) {
            if (this._start_point !== null) {
                this._current_point = pos;
                this.update_box();
            }
        };
        ;
        DrawBoxTool.prototype.on_switch_out = function (pos) {
            if (this._start_point !== null) {
                this._current_point = null;
                this.update_box();
            }
        };
        ;
        DrawBoxTool.prototype.on_cancel = function (pos) {
            if (this.entity !== null) {
                if (this._start_point !== null) {
                    this.destroy_entity();
                    this._start_point = null;
                }
            }
            else {
                this._view.unselect_all_entities();
                this._view.view.set_current_tool(new labelling_tool.SelectEntityTool(this._view));
            }
            return true;
        };
        ;
        DrawBoxTool.prototype.on_left_click = function (pos, event) {
            if (this._start_point === null) {
                this._start_point = pos;
                this._current_point = pos;
                this.create_entity(pos);
            }
            else {
                this._current_point = pos;
                this.update_box();
                this.entity.commit();
                this._start_point = null;
                this._current_point = null;
            }
        };
        ;
        DrawBoxTool.prototype.on_move = function (pos) {
            if (this._start_point !== null) {
                this._current_point = pos;
                this.update_box();
            }
        };
        ;
        DrawBoxTool.prototype.create_entity = function (pos) {
            var model = new_BoxLabelModel(pos, { x: 0.0, y: 0.0 });
            var entity = this._view.get_or_create_entity_for_model(model);
            this.entity = entity;
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this._view.add_child(entity);
            this._view.select_entity(entity, false, false);
            this._view.view.thaw();
        };
        ;
        DrawBoxTool.prototype.destroy_entity = function () {
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this.entity.destroy();
            this.entity = null;
            this._view.view.thaw();
        };
        ;
        DrawBoxTool.prototype.update_box = function () {
            if (this.entity !== null) {
                var box = null;
                if (this._start_point !== null) {
                    if (this._current_point !== null) {
                        box = labelling_tool.AABox_from_points([this._start_point, this._current_point]);
                    }
                    else {
                        box = new labelling_tool.AABox(this._start_point, this._start_point);
                    }
                }
                this.entity.model.centre = box.centre();
                this.entity.model.size = box.size();
                this.entity.update();
            }
        };
        ;
        return DrawBoxTool;
    }(labelling_tool.AbstractTool));
    labelling_tool.DrawBoxTool = DrawBoxTool;
})(labelling_tool || (labelling_tool = {}));
