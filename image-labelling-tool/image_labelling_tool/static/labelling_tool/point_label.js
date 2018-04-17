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
/// <reference path="./root_label_view.ts" />
var labelling_tool;
(function (labelling_tool) {
    function new_PointLabelModel(position) {
        return { label_type: 'point', label_class: null, position: position };
    }
    /*
    Point label entity
     */
    var PointLabelEntity = (function (_super) {
        __extends(PointLabelEntity, _super);
        function PointLabelEntity(view, model) {
            return _super.call(this, view, model) || this;
        }
        PointLabelEntity.prototype.attach = function () {
            _super.prototype.attach.call(this);
            this.circle = this.root_view.world.append("circle")
                .attr('r', 4.0);
            this.update();
            var self = this;
            this.circle.on("mouseover", function () {
                self._on_mouse_over_event();
            }).on("mouseout", function () {
                self._on_mouse_out_event();
            });
            this._update_style();
        };
        PointLabelEntity.prototype.detach = function () {
            this.circle.remove();
            this.circle = null;
            _super.prototype.detach.call(this);
        };
        PointLabelEntity.prototype._on_mouse_over_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_in(this);
            }
        };
        PointLabelEntity.prototype._on_mouse_out_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_out(this);
            }
        };
        PointLabelEntity.prototype.update = function () {
            var position = this.model.position;
            this.circle
                .attr('cx', position.x)
                .attr('cy', position.y);
        };
        PointLabelEntity.prototype.commit = function () {
            this.root_view.commit_model(this.model);
        };
        PointLabelEntity.prototype._update_style = function () {
            if (this._attached) {
                var stroke_colour = this._outline_colour();
                if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.HIDDEN) {
                    this.circle.attr("visibility", "hidden");
                }
                else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FAINT) {
                    stroke_colour = stroke_colour.with_alpha(0.2);
                    this.circle.attr("style", "fill:none;stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.circle.attr("visibility", "visible");
                }
                else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FULL) {
                    var circle_fill_colour = this.root_view.view.colour_for_label_class(this.model.label_class);
                    if (this._hover) {
                        circle_fill_colour = circle_fill_colour.lighten(0.4);
                    }
                    circle_fill_colour = circle_fill_colour.with_alpha(0.35);
                    stroke_colour = stroke_colour.with_alpha(0.5);
                    this.circle.attr("style", "fill:" + circle_fill_colour.to_rgba_string() + ";stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.circle.attr("visibility", "visible");
                }
            }
        };
        PointLabelEntity.prototype.compute_centroid = function () {
            return this.model.position;
        };
        PointLabelEntity.prototype.compute_bounding_box = function () {
            var centre = this.compute_centroid();
            return new labelling_tool.AABox({ x: centre.x - 1, y: centre.y - 1 }, { x: centre.x + 1, y: centre.y + 1 });
        };
        PointLabelEntity.prototype.contains_pointer_position = function (point) {
            return labelling_tool.compute_sqr_dist(point, this.model.position) <= (4.0 * 4.0);
        };
        return PointLabelEntity;
    }(labelling_tool.AbstractLabelEntity));
    labelling_tool.PointLabelEntity = PointLabelEntity;
    labelling_tool.register_entity_factory('point', function (root_view, model) {
        return new PointLabelEntity(root_view, model);
    });
    /*
    Draw point tool
     */
    var DrawPointTool = (function (_super) {
        __extends(DrawPointTool, _super);
        function DrawPointTool(view, entity) {
            var _this = _super.call(this, view) || this;
            _this.entity = entity;
            return _this;
        }
        DrawPointTool.prototype.on_init = function () {
        };
        ;
        DrawPointTool.prototype.on_shutdown = function () {
        };
        ;
        // on_switch_in(pos: Vector2) {
        //     if (this.entity !== null) {
        //         this.add_point(pos);
        //     }
        // };
        //
        // on_switch_out(pos: Vector2) {
        //     if (this.entity !== null) {
        //         this.remove_last_point();
        //     }
        // };
        //
        DrawPointTool.prototype.on_cancel = function (pos) {
            this._view.unselect_all_entities();
            this._view.view.set_current_tool(new labelling_tool.SelectEntityTool(this._view));
            return true;
        };
        ;
        DrawPointTool.prototype.on_left_click = function (pos, event) {
            this.create_entity(pos);
            this.entity.update();
            this.entity.commit();
        };
        ;
        DrawPointTool.prototype.create_entity = function (position) {
            var model = new_PointLabelModel(position);
            var entity = this._view.get_or_create_entity_for_model(model);
            this.entity = entity;
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this._view.add_child(entity);
            this._view.select_entity(entity, false, false);
            this._view.view.thaw();
        };
        ;
        DrawPointTool.prototype.destroy_entity = function () {
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this.entity.destroy();
            this.entity = null;
            this._view.view.thaw();
        };
        ;
        return DrawPointTool;
    }(labelling_tool.AbstractTool));
    labelling_tool.DrawPointTool = DrawPointTool;
})(labelling_tool || (labelling_tool = {}));
