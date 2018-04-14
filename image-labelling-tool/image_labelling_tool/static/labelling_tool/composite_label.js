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
var labelling_tool;
(function (labelling_tool) {
    function new_CompositeLabelModel() {
        return { label_type: 'composite', label_class: null, components: [] };
    }
    labelling_tool.new_CompositeLabelModel = new_CompositeLabelModel;
    /*
    Composite label entity
     */
    var CompositeLabelEntity = (function (_super) {
        __extends(CompositeLabelEntity, _super);
        function CompositeLabelEntity(view, model) {
            var _this = _super.call(this, view, model) || this;
            _this._centroid = null;
            return _this;
        }
        CompositeLabelEntity.prototype.attach = function () {
            _super.prototype.attach.call(this);
            this.circle = this.root_view.world.append("circle")
                .attr('r', 8.0);
            this.central_circle = this.root_view.world.append("circle")
                .attr("pointer-events", "none")
                .attr('r', 4.0);
            this.shape_line = d3.svg.line()
                .x(function (d) { return d.x; })
                .y(function (d) { return d.y; })
                .interpolate("linear-closed");
            this.connections_group = null;
            this.update();
            var self = this;
            this.circle.on("mouseover", function () {
                self._on_mouse_over_event();
            }).on("mouseout", function () {
                self._on_mouse_out_event();
            });
            this._update_style();
        };
        CompositeLabelEntity.prototype.detach = function () {
            this.circle.remove();
            this.central_circle.remove();
            this.connections_group.remove();
            this.circle = null;
            this.central_circle = null;
            this.shape_line = null;
            this.connections_group = null;
            _super.prototype.detach.call(this);
        };
        CompositeLabelEntity.prototype._on_mouse_over_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_in(this);
            }
        };
        CompositeLabelEntity.prototype._on_mouse_out_event = function () {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_out(this);
            }
        };
        CompositeLabelEntity.prototype.update = function () {
            var component_centroids = this._compute_component_centroids();
            this._centroid = labelling_tool.compute_centroid_of_points(component_centroids);
            this.circle
                .attr('cx', this._centroid.x)
                .attr('cy', this._centroid.y);
            this.central_circle
                .attr('cx', this._centroid.x)
                .attr('cy', this._centroid.y);
            if (this.connections_group !== null) {
                this.connections_group.remove();
                this.connections_group = null;
            }
            this.connections_group = this.root_view.world.append("g");
            for (var i = 0; i < component_centroids.length; i++) {
                this.connections_group.append("path")
                    .attr("d", this.shape_line([this._centroid, component_centroids[i]]))
                    .attr("stroke-width", 1)
                    .attr("stroke-dasharray", "3, 3")
                    .attr("style", "stroke:rgba(255,0,255,0.6);");
                this.connections_group.append("circle")
                    .attr("cx", component_centroids[i].x)
                    .attr("cy", component_centroids[i].y)
                    .attr("r", 3)
                    .attr("stroke-width", 1)
                    .attr("style", "stroke:rgba(255,0,255,0.6);fill: rgba(255,0,255,0.25);");
            }
        };
        CompositeLabelEntity.prototype.commit = function () {
            this.root_view.commit_model(this.model);
        };
        CompositeLabelEntity.prototype._update_style = function () {
            if (this._attached) {
                var stroke_colour = this._outline_colour();
                if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FAINT) {
                    stroke_colour = stroke_colour.with_alpha(0.2);
                    this.circle.attr("style", "fill:none;stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.connections_group.selectAll("path")
                        .attr("style", "stroke:rgba(255,0,255,0.2);");
                    this.connections_group.selectAll("circle")
                        .attr("style", "stroke:rgba(255,0,255,0.2);fill: none;");
                }
                else if (this.root_view.view.label_visibility == labelling_tool.LabelVisibility.FULL) {
                    var circle_fill_colour = new labelling_tool.Colour4(255, 128, 255, 1.0);
                    var central_circle_fill_colour = this.root_view.view.colour_for_label_class(this.model.label_class);
                    var connection_fill_colour = new labelling_tool.Colour4(255, 0, 255, 1.0);
                    var connection_stroke_colour = new labelling_tool.Colour4(255, 0, 255, 1.0);
                    if (this._hover) {
                        circle_fill_colour = circle_fill_colour.lighten(0.4);
                        central_circle_fill_colour = central_circle_fill_colour.lighten(0.4);
                        connection_fill_colour = connection_fill_colour.lighten(0.4);
                        connection_stroke_colour = connection_stroke_colour.lighten(0.4);
                    }
                    circle_fill_colour = circle_fill_colour.with_alpha(0.35);
                    central_circle_fill_colour = central_circle_fill_colour.with_alpha(0.35);
                    connection_fill_colour = connection_fill_colour.with_alpha(0.25);
                    connection_stroke_colour = connection_stroke_colour.with_alpha(0.6);
                    stroke_colour = stroke_colour.with_alpha(0.5);
                    this.circle.attr("style", "fill:" + circle_fill_colour.to_rgba_string() + ";stroke:" + connection_stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.central_circle.attr("style", "fill:" + central_circle_fill_colour.to_rgba_string() + ";stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.connections_group.selectAll("path")
                        .attr("style", "stroke:rgba(255,0,255,0.6);");
                    this.connections_group.selectAll("circle")
                        .attr("style", "stroke:" + connection_stroke_colour.to_rgba_string() + ";fill:" + connection_fill_colour.to_rgba_string() + ";");
                }
            }
        };
        CompositeLabelEntity.prototype._compute_component_centroids = function () {
            var component_centroids = [];
            for (var i = 0; i < this.model.components.length; i++) {
                var model_id = this.model.components[i];
                var entity = this.root_view.get_entity_for_model_id(model_id);
                var centroid = entity.compute_centroid();
                component_centroids.push(centroid);
            }
            return component_centroids;
        };
        CompositeLabelEntity.prototype.compute_centroid = function () {
            return this._centroid;
        };
        ;
        CompositeLabelEntity.prototype.compute_bounding_box = function () {
            var centre = this.compute_centroid();
            return new labelling_tool.AABox({ x: centre.x - 1, y: centre.y - 1 }, { x: centre.x + 1, y: centre.y + 1 });
        };
        CompositeLabelEntity.prototype.contains_pointer_position = function (point) {
            return labelling_tool.compute_sqr_dist(point, this._centroid) <= (8.0 * 8.0);
        };
        CompositeLabelEntity.prototype.notify_model_destroyed = function (model_id) {
            var index = this.model.components.indexOf(model_id);
            if (index !== -1) {
                // Remove the model ID from the components array
                this.model.components = this.model.components.slice(0, index).concat(this.model.components.slice(index + 1));
                this.update();
            }
        };
        return CompositeLabelEntity;
    }(labelling_tool.AbstractLabelEntity));
    labelling_tool.CompositeLabelEntity = CompositeLabelEntity;
    labelling_tool.register_entity_factory('composite', function (root_view, model) {
        return new CompositeLabelEntity(root_view, model);
    });
})(labelling_tool || (labelling_tool = {}));
