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

/// <reference path="./math_primitives.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./abstract_tool.ts" />
/// <reference path="./select_tools.ts" />
/// <reference path="./root_label_view.ts" />

module labelling_tool {
    /*
    Point label model
     */
    interface PointLabelModel extends AbstractLabelModel {
        position: Vector2;
    }

    function new_PointLabelModel(position: Vector2): PointLabelModel {
        return {label_type: 'point', label_class: null, position: position};
    }


    /*
    Point label entity
     */
    export class PointLabelEntity extends AbstractLabelEntity<PointLabelModel> {
        circle: any;
        connections_group: any;

        constructor(view: RootLabelView, model: PointLabelModel) {
            super(view, model);
        }

        attach() {
            super.attach();
            this.circle = this.root_view.world.append("circle")
                .attr('r', 4.0);

            this.update();

            var self = this;
            this.circle.on("mouseover", function() {
                self._on_mouse_over_event();
            }).on("mouseout", function() {
                self._on_mouse_out_event();
            });


            this._update_style();
        }

        detach() {
            this.circle.remove();
            this.circle = null;
            super.detach();
        }


        _on_mouse_over_event() {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_in(this);
            }
        }

        _on_mouse_out_event() {
            for (var i = 0; i < this._event_listeners.length; i++) {
                this._event_listeners[i].on_mouse_out(this);
            }
        }


        update() {
            var position = this.model.position;

            this.circle
                .attr('cx', position.x)
                .attr('cy', position.y);
        }

        commit() {
            this.root_view.commit_model(this.model);
        }


        _update_style() {
            if (this._attached) {
                var stroke_colour: Colour4 = this._outline_colour();

                if (this.root_view.view.label_visibility == LabelVisibility.HIDDEN) {
                    this.circle.attr("visibility", "hidden");
                }
                else if (this.root_view.view.label_visibility == LabelVisibility.FAINT) {
                    stroke_colour = stroke_colour.with_alpha(0.2);
                    this.circle.attr("style", "fill:none;stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this.circle.attr("visibility", "visible");
                }
                else if (this.root_view.view.label_visibility == LabelVisibility.FULL) {
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
        }

        compute_centroid(): Vector2 {
            return this.model.position;
        }

        compute_bounding_box(): AABox {
            var centre = this.compute_centroid();
            return new AABox({x: centre.x - 1, y: centre.y - 1}, {x: centre.x + 1, y: centre.y + 1});
        }

        contains_pointer_position(point: Vector2): boolean {
            return compute_sqr_dist(point, this.model.position) <= (4.0 * 4.0);
        }
    }


    register_entity_factory('point', (root_view: RootLabelView, model: AbstractLabelModel) => {
            return new PointLabelEntity(root_view, model as PointLabelModel);
    });


    /*
    Draw point tool
     */
    export class DrawPointTool extends AbstractTool {
        entity: PointLabelEntity;

        constructor(view: RootLabelView, entity: PointLabelEntity) {
            super(view);
            this.entity = entity;
        }

        on_init() {
        };

        on_shutdown() {
        };

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
        on_cancel(pos: Vector2): boolean {
            this._view.unselect_all_entities();
            this._view.view.set_current_tool(new SelectEntityTool(this._view));
            return true;
        };

        on_left_click(pos: Vector2, event: any) {
            this.create_entity(pos);
            this.entity.update();
            this.entity.commit();
        };


        create_entity(position: Vector2) {
            var model = new_PointLabelModel(position);
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
    }
}
