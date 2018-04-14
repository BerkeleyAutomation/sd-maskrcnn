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

/// <reference path="../polyk.d.ts" />
/// <reference path="./math_primitives.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./abstract_tool.ts" />
/// <reference path="./select_tools.ts" />

module labelling_tool {
    /*
    Polygonal label model
     */
    interface PolygonalLabelModel extends AbstractLabelModel {
        vertices: Vector2[];
    }

    function new_PolygonalLabelModel(): PolygonalLabelModel {
        return {label_type: 'polygon', label_class: null, vertices: []};
    }



    /*
    Polygonal label entity
     */
    export class PolygonalLabelEntity extends AbstractLabelEntity<PolygonalLabelModel> {
        _polyk_poly: number[];
        _centroid: Vector2;
        _bounding_box: AABox;
        poly: any;
        shape_line: any;


        constructor(view: RootLabelView, model: PolygonalLabelModel) {
            super(view, model);
            this._polyk_poly = [];
            this._centroid = null;
            this._bounding_box = null;
            this.poly = null;
            this.shape_line = null;
        }

        attach() {
            super.attach();

            this.shape_line = d3.svg.line()
                .x(function (d: any) { return d.x; })
                .y(function (d: any) { return d.y; })
                .interpolate("linear-closed");

            this.poly = this.root_view.world.append("path");
            this.poly.data(this.model.vertices).attr("d", this.shape_line(this.model.vertices));

            this.poly.on("mouseover", () => {
                for (var i = 0; i < this._event_listeners.length; i++) {
                    this._event_listeners[i].on_mouse_in(this);
                }
            });

            this.poly.on("mouseout", () => {
                for (var i = 0; i < this._event_listeners.length; i++) {
                    this._event_listeners[i].on_mouse_out(this);
                }
            });

            this._update_polyk_poly();
            this._update_style();
        };

        detach() {
            this.poly.remove();
            this.poly = null;
            this.shape_line = null;
            this._polyk_poly = [];
            super.detach();
        };

        _update_polyk_poly() {
            this._polyk_poly = [];
            for (var i = 0; i < this.model.vertices.length; i++) {
                this._polyk_poly.push(this.model.vertices[i].x);
                this._polyk_poly.push(this.model.vertices[i].y);
            }
        }

        update() {
            this.poly.data(this.model.vertices).attr("d", this.shape_line(this.model.vertices));
            this._update_polyk_poly();
            this._centroid = null;
            this._bounding_box = null;
        }

        commit() {
            this.root_view.commit_model(this.model);
        }


        _update_style() {
            if (this._attached) {
                var stroke_colour: Colour4 = this._outline_colour();

                if (this.root_view.view.label_visibility == LabelVisibility.HIDDEN) {
                    this.poly.attr("visibility", "hidden");
                }
                else {
                    var fill_colour: Colour4 = this.root_view.view.colour_for_label_class(this.model.label_class);

                    if (this.root_view.view.label_visibility == LabelVisibility.FAINT) {
                        stroke_colour = stroke_colour.with_alpha(0.2);

                        if (this._hover) {
                            fill_colour = fill_colour.lighten(0.4);
                        }
                        if (this._selected) {
                            fill_colour = fill_colour.lerp(new Colour4(255, 128, 0.0, 1.0), 0.2);
                        }
                        fill_colour = fill_colour.with_alpha(0.1);
                    }
                    else if (this.root_view.view.label_visibility == LabelVisibility.FULL) {
                        if (this._hover) {
                            fill_colour = fill_colour.lighten(0.4);
                        }
                        if (this._selected) {
                            fill_colour = fill_colour.lerp(new Colour4(255, 128, 0.0, 1.0), 0.2);
                        }
                        fill_colour = fill_colour.with_alpha(0.35);
                    }

                    this.poly.attr("style", "fill:" + fill_colour.to_rgba_string() + ";stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1")
                        .attr("visibility", "visible");
                }
            }
        }

        compute_centroid(): Vector2 {
            if (this._centroid === null) {
                this._centroid = compute_centroid_of_points(this.model.vertices);
            }
            return this._centroid;
        }

        compute_bounding_box(): AABox {
            if (this._bounding_box === null) {
                this._bounding_box = AABox_from_points(this.model.vertices);
            }
            return this._bounding_box;
        }

        contains_pointer_position(point: Vector2): boolean {
            if (this.compute_bounding_box().contains_point(point)) {
                return PolyK.ContainsPoint(this._polyk_poly, point.x, point.y);
            }
            else {
                return false;
            }
        }

        distance_to_point(point: Vector2): number {
            if (PolyK.ContainsPoint(this._polyk_poly, point.x, point.y)) {
                return 0.0;
            }
            else {
                var e = PolyK.ClosestEdge(this._polyk_poly, point.x, point.y);
                return e.dist;
            }
        }
    }


    register_entity_factory('polygon', (root_view: RootLabelView, model: AbstractLabelModel) => {
        return new PolygonalLabelEntity(root_view, model as PolygonalLabelModel);
    });


    /*
    Draw polygon tool
     */
    export class DrawPolygonTool extends AbstractTool {
        entity: PolygonalLabelEntity;
        _last_vertex_marker: any;
        _last_vertex_marker_visible: boolean;
        _key_event_listener: (event: any)=>any;

        constructor(view: RootLabelView, entity: PolygonalLabelEntity) {
            super(view);
            var self = this;
            this.entity = entity;
            this._last_vertex_marker = null;
            this._key_event_listener = function(event) {
                self.on_key_press(event);
            }
        }

        on_init() {
            this._last_vertex_marker = this._view.world.append("circle");
            this._last_vertex_marker.attr("r", "3.0");
            this._last_vertex_marker.attr("visibility", "hidden");
            this._last_vertex_marker.style("fill", "rgba(128,0,192,0.1)");
            this._last_vertex_marker.style("stroke-width", "1.0");
            this._last_vertex_marker.style("stroke", "rgba(192,0,255,1.0)");
            this._last_vertex_marker_visible = false;
        };

        on_shutdown() {
            this._last_vertex_marker.remove();
            this._last_vertex_marker = null;
        };

        on_switch_in(pos: Vector2) {
            if (this.entity !== null) {
                this.add_point(pos);
                this._last_vertex_marker_visible = true;
            }
            document.addEventListener("keypress", this._key_event_listener);
        };

        on_switch_out(pos: Vector2) {
            this._last_vertex_marker_visible = false;
            if (this.entity !== null) {
                this.remove_last_point();
                this.entity.commit();
            }
            document.removeEventListener("keypress", this._key_event_listener);
        };

        on_key_press(event: any) {
            var key: string = event.key;
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
        }


        on_cancel(pos: Vector2): boolean {
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
                this._view.view.set_current_tool(new SelectEntityTool(this._view));
            }
            return true;
        };

        on_left_click(pos: Vector2, event: any) {
            this.add_point(pos);
        };

        on_move(pos: Vector2) {
            this.update_last_point(pos);
        };



        create_entity() {
            var model = new_PolygonalLabelModel();
            var entity = this._view.get_or_create_entity_for_model(model);
            this.entity = entity;
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this._view.add_child(entity);
            this._view.select_entity(entity, false, false);
            this._view.view.thaw();
        };

        destroy_entity() {
            // Freeze to prevent this temporary change from being sent to the backend
            this._view.view.freeze();
            this.entity.destroy();
            this.entity = null;
            this._view.view.thaw();
        };

        get_vertices() {
            return this.entity !== null  ?  this.entity.model.vertices  :  null;
        };

        update_poly() {
            var last_vertex_pos: Vector2 = null;
            if (this.entity !== null) {
                this.entity.update();
                var vertices = this.get_vertices();
                if (vertices.length >= 2 && this._last_vertex_marker_visible) {
                    var last_vertex_pos = vertices[vertices.length - 2];
                }
            }
            this.show_last_vertex_at(last_vertex_pos);
        };

        show_last_vertex_at(pos: Vector2) {
            if (pos === null) {
                this._last_vertex_marker.attr("visibility", "hidden");
            }
            else {
                this._last_vertex_marker.attr("visibility", "visible");
                this._last_vertex_marker.attr("cx", pos.x);
                this._last_vertex_marker.attr("cy", pos.y);
            }
        }


        add_point(pos: Vector2) {
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

        update_last_point(pos: Vector2) {
            var vertices = this.get_vertices();
            if (vertices !== null) {
                vertices[vertices.length - 1] = pos;
                this.update_poly();
            }
        };

        remove_last_point() {
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
    }
}
