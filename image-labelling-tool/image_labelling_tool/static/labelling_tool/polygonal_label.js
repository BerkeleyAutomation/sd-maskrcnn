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
/// <reference path="../polyk.d.ts" />
/// <reference path="./math_primitives.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./abstract_tool.ts" />
/// <reference path="./select_tools.ts" />
var labelling_tool;
(function (labelling_tool) {
    function new_PolygonalLabelModel() {
        return { label_type: 'polygon', label_class: null, vertices: [] };
    }
    /*
    Polygonal label entity
     */
    var PolygonalLabelEntity = (function (_super) {
        __extends(PolygonalLabelEntity, _super);
        function PolygonalLabelEntity(view, model) {
            var _this = _super.call(this, view, model) || this;
            _this._polyk_poly = [];
            _this._centroid = null;
            _this._bounding_box = null;
            _this.poly = null;
            _this.shape_line = null;
            return _this;
        }
        PolygonalLabelEntity.prototype.attach = function () {
            var _this = this;
            _super.prototype.attach.call(this);
            this.shape_line = d3.svg.line()
                .x(function (d) { return d.x; })
                .y(function (d) { return d.y; })
                .interpolate("linear-closed");
            this.poly = this.root_view.world.append("path");
            this.poly.data(this.model.vertices).attr("d", this.shape_line(this.model.vertices));
            this.poly.on("mouseover", function () {
                for (var i = 0; i < _this._event_listeners.length; i++) {
                    _this._event_listeners[i].on_mouse_in(_this);
                }
            });
            this.poly.on("mouseout", function () {
                for (var i = 0; i < _this._event_listeners.length; i++) {
                    _this._event_listeners[i].on_mouse_out(_this);
                }
            });
            this._update_polyk_poly();
            this._update_style();
        };
        ;
        PolygonalLabelEntity.prototype.detach = function () {
            this.poly.remove();
            this.poly = null;
            this.shape_line = null;
            this._polyk_poly = [];
            _super.prototype.detach.call(this);
        };
        ;
        PolygonalLabelEntity.prototype._update_polyk_poly = function () {
            this._polyk_poly = [];
            for (var i = 0; i < this.model.vertices.length; i++) {
                this._polyk_poly.push(this.model.vertices[i].x);
                this._polyk_poly.push(this.model.vertices[i].y);
            }
        };
        PolygonalLabelEntity.prototype.update = function () {
            this.poly.data(this.model.vertices).attr("d", this.shape_line(this.model.vertices));
            this._update_polyk_poly();
            this._centroid = null;
            this._bounding_box = null;
        };
        PolygonalLabelEntity.prototype.commit = function () {
            this.root_view.commit_model(this.model);
        };
        PolygonalLabelEntity.prototype._update_style = function () {
            if (this._attached) {
                var stroke_colour = this._outline_colour();
                if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.HIDDEN) {
                    this.poly.attr("visibility", "hidden");
                }
                else {
                    var fill_colour = this.root_view.view.colour_for_label_class(this.model.label_class);
                    if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FAINT) {
                        stroke_colour = stroke_colour.with_alpha(0.2);
                        if (this._hover) {
                            fill_colour = fill_colour.lighten(0.4);
                        }
                        if (this._selected) {
                            fill_colour = fill_colour.lerp(new labelling_tool.Colour4(255, 128, 0.0, 1.0), 0.2);
                        }
                        fill_colour = fill_colour.with_alpha(0.1);
                    }
                    else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FULL) {
                        if (this._hover) {
                            fill_colour = fill_colour.lighten(0.4);
                        }
                        if (this._selected) {
                            fill_colour = fill_colour.lerp(new labelling_tool.Colour4(255, 128, 0.0, 1.0), 0.2);
                        }
                        fill_colour = fill_colour.with_alpha(0.35);
                    }
                    this.poly.attr("style", "fill:" + fill_colour.to_rgba_string() + ";stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1")
                        .attr("visibility", "visible");
                }
            }
        };
        PolygonalLabelEntity.prototype.compute_centroid = function () {
            if (this._centroid === null) {
                this._centroid = labelling_tool.compute_centroid_of_points(this.model.vertices);
            }
            return this._centroid;
        };
        PolygonalLabelEntity.prototype.compute_bounding_box = function () {
            if (this._bounding_box === null) {
                this._bounding_box = labelling_tool.AABox_from_points(this.model.vertices);
            }
            return this._bounding_box;
        };
        PolygonalLabelEntity.prototype.contains_pointer_position = function (point) {
            if (this.compute_bounding_box().contains_point(point)) {
                return PolyK.ContainsPoint(this._polyk_poly, point.x, point.y);
            }
            else {
                return false;
            }
        };
        PolygonalLabelEntity.prototype.distance_to_point = function (point) {
            if (PolyK.ContainsPoint(this._polyk_poly, point.x, point.y)) {
                return 0.0;
            }
            else {
                var e = PolyK.ClosestEdge(this._polyk_poly, point.x, point.y);
                return e.dist;
            }
        };
        return PolygonalLabelEntity;
    }(labelling_tool.AbstractLabelEntity));
    labelling_tool.PolygonalLabelEntity = PolygonalLabelEntity;
    labelling_tool.register_entity_factory('polygon', function (root_view, model) {
        return new PolygonalLabelEntity(root_view, model);
    });
    /*
    Draw polygon tool
     */
    var DrawPolygonTool = (function (_super) {
        __extends(DrawPolygonTool, _super);
        function DrawPolygonTool(view, entity) {
            var _this = _super.call(this, view) || this;
            var self = _this;
            _this.entity = entity;
            _this._last_vertex_marker = null;
            _this._key_event_listener = function (event) {
                self.on_key_press(event);
            };
            return _this;
        }
        DrawPolygonTool.prototype.on_init = function () {
            this._last_vertex_marker = this._view.world.append("circle");
            this._last_vertex_marker.attr("r", "3.0");
            this._last_vertex_marker.attr("visibility", "hidden");
            this._last_vertex_marker.style("fill", "rgba(128,0,192,0.1)");
            this._last_vertex_marker.style("stroke-width", "1.0");
            this._last_vertex_marker.style("stroke", "rgba(192,0,255,1.0)");
            this._last_vertex_marker_visible = false;
        };
        ;
        DrawPolygonTool.prototype.on_shutdown = function () {
            this._last_vertex_marker.remove();
            this._last_vertex_marker = null;
        };
        ;
        DrawPolygonTool.prototype.on_switch_in = function (pos) {
            if (this.entity !== null) {
                this.add_point(pos);
                this._last_vertex_marker_visible = true;
            }
            document.addEventListener("keypress", this._key_event_listener);
        };
        ;
        DrawPolygonTool.prototype.on_switch_out = function (pos) {
            this._last_vertex_marker_visible = false;
            if (this.entity !== null) {
                this.remove_last_point();
                this.entity.commit();
            }
            document.removeEventListener("keypress", this._key_event_listener);
        };
        ;
        DrawPolygonTool.prototype.on_key_press = function (event) {
            var key = event.key;
            if (key === ',') {
                // Shift vertices back
                var vertices = this.get_vertices();
                if (vertices !== null && vertices.length >= 3) {
                    var last_vertex = vertices[vertices.length - 2];
                    // Remove the last vertex
                    vertices.splice(vertices.length - 2, 1);
                    // Insert the last vertex at the beginning
                    vertices.splice(0, 0, last_vertex);
                    this.update_poly();
                    this.entity.commit();
                }
            }
            else if (key == '.') {
                // Shift vertices forward
                var vertices = this.get_vertices();
                if (vertices !== null && vertices.length >= 3) {
                    var first_vertex = vertices[0];
                    // Remove the first vertex
                    vertices.splice(0, 1);
                    // Insert the first vertex at the end, before the current vertex
                    vertices.splice(vertices.length - 1, 0, first_vertex);
                    this.update_poly();
                    this.entity.commit();
                }
            }
            else if (key == '/') {
                var vertices = this.get_vertices();
                if (vertices !== null && vertices.length >= 3) {
                    // Remove the last vertex
                    vertices.splice(vertices.length - 2, 1);
                    this.update_poly();
                    this.entity.commit();
                }
            }
        };
        DrawPolygonTool.prototype.on_cancel = function (pos) {
            if (this.entity !== null) {
                this._last_vertex_marker_visible = false;
                this.remove_last_point();
                var vertices = this.get_vertices();
                if (vertices.length == 1) {
                    this.destroy_entity();
                }
                else {
                    this.entity.commit();
                    this.entity = null;
                }
            }
            else {
                this._view.unselect_all_entities();
                this._view.view.set_current_tool(new labelling_tool.SelectEntityTool(this._view));
            }
            return true;
        };
        ;
        DrawPolygonTool.prototype.on_left_click = function (pos, event) {
            this.add_point(pos);
        };
        ;
        DrawPolygonTool.prototype.on_move = function (pos) {
            this.update_last_point(pos);
        };
        ;
        DrawPolygonTool.prototype.create_entity = function () {
            var model = new_PolygonalLabelModel();
            var entity = this._view.get_or_create_entity_for_model(model);
            this.entity = entity;
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this._view.add_child(entity);
            this._view.select_entity(entity, false, false);
            this._view.view.thaw();
        };
        ;
        DrawPolygonTool.prototype.destroy_entity = function () {
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this.entity.destroy();
            this.entity = null;
            this._view.view.thaw();
        };
        ;
        DrawPolygonTool.prototype.get_vertices = function () {
            return this.entity !== null ? this.entity.model.vertices : null;
        };
        ;
        DrawPolygonTool.prototype.update_poly = function () {
            var last_vertex_pos = null;
            if (this.entity !== null) {
                this.entity.update();
                var vertices = this.get_vertices();
                if (vertices.length >= 2 && this._last_vertex_marker_visible) {
                    var last_vertex_pos = vertices[vertices.length - 2];
                }
            }
            this.show_last_vertex_at(last_vertex_pos);
        };
        ;
        DrawPolygonTool.prototype.show_last_vertex_at = function (pos) {
            if (pos === null) {
                this._last_vertex_marker.attr("visibility", "hidden");
            }
            else {
                this._last_vertex_marker.attr("visibility", "visible");
                this._last_vertex_marker.attr("cx", pos.x);
                this._last_vertex_marker.attr("cy", pos.y);
            }
        };
        DrawPolygonTool.prototype.add_point = function (pos) {
            var entity_is_new = false;
            if (this.entity === null) {
                this.create_entity();
                entity_is_new = true;
            }
            var vertices = this.get_vertices();
            if (entity_is_new) {
                // Add a duplicate vertex; this second vertex will follow the mouse
                vertices.push(pos);
            }
            vertices.push(pos);
            this.update_poly();
        };
        ;
        DrawPolygonTool.prototype.update_last_point = function (pos) {
            var vertices = this.get_vertices();
            if (vertices !== null) {
                vertices[vertices.length - 1] = pos;
                this.update_poly();
            }
        };
        ;
        DrawPolygonTool.prototype.remove_last_point = function () {
            var vertices = this.get_vertices();
            if (vertices !== null) {
                if (vertices.length > 0) {
                    vertices.splice(vertices.length - 1, 1);
                    this.update_poly();
                }
                if (vertices.length === 0) {
                    this.destroy_entity();
                }
            }
        };
        ;
        return DrawPolygonTool;
    }(labelling_tool.AbstractTool));
    labelling_tool.DrawPolygonTool = DrawPolygonTool;
})(labelling_tool || (labelling_tool = {}));
