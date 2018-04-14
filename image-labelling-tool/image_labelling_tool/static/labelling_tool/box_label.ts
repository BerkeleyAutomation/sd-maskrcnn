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

module labelling_tool {
    /*
    Box label model
     */
    interface BoxLabelModel extends AbstractLabelModel {
        centre: Vector2;
        size: Vector2;
    }

    function new_BoxLabelModel(centre: Vector2, size: Vector2): BoxLabelModel {
        return {label_type: 'box', label_class: null, centre: centre, size: size};
    }

    function BoxLabel_box(label: BoxLabelModel): AABox {
        var lower = {x: label.centre.x - label.size.x*0.5, y: label.centre.y - label.size.y*0.5};
        var upper = {x: label.centre.x + label.size.x*0.5, y: label.centre.y + label.size.y*0.5};
        return new AABox(lower, upper);
    }


    /*
    Box label entity
     */
    export class BoxLabelEntity extends AbstractLabelEntity<BoxLabelModel> {
        _rect: any;


        constructor(view: RootLabelView, model: BoxLabelModel) {
            super(view, model);
        }


        attach() {
            super.attach();

            this._rect = this.root_view.world.append("rect")
                .attr("x", 0).attr("y", 0)
                .attr("width", 0).attr("height", 0);

            this.update();

            var self = this;
            this._rect.on("mouseover", function() {
                self._on_mouse_over_event();
            }).on("mouseout", function() {
                self._on_mouse_out_event();
            });


            this._update_style();
        };

        detach() {
            this._rect.remove();
            this._rect = null;
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
            var box = BoxLabel_box(this.model);
            var size = box.size();

            this._rect
                .attr('x', box.lower.x).attr('y', box.lower.y)
                .attr('width', size.x).attr('height', size.y);
        }

        commit() {
            this.root_view.commit_model(this.model);
        }

        _update_style() {
            if (this._attached) {
                var stroke_colour: Colour4 = this._outline_colour();

                if (this.root_view.view.label_visibility == LabelVisibility.HIDDEN) {
                    this._rect.attr("visibility", "hidden");
                }
                else if (this.root_view.view.label_visibility == LabelVisibility.FAINT) {
                    stroke_colour = stroke_colour.with_alpha(0.2);
                    this._rect.attr("style", "fill:none;stroke:" + stroke_colour.to_rgba_string() + ";stroke-width:1");
                    this._rect.attr("visibility", "visible");
                }
                else if (this.root_view.view.label_visibility == LabelVisibility.FULL) {
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
        }

        compute_centroid(): Vector2 {
            return this.model.centre;
        };

        compute_bounding_box(): AABox {
            return BoxLabel_box(this.model);
        };

        contains_pointer_position(point: Vector2): boolean {
            return this.compute_bounding_box().contains_point(point);
        }

        distance_to_point(point: Vector2): number {
            return BoxLabel_box(this.model).distance_to(point);
        }
    }


    register_entity_factory('box', (root_view: RootLabelView, model: AbstractLabelModel) => {
            return new BoxLabelEntity(root_view, model as BoxLabelModel);
    });


    /*
    Draw box tool
     */
    export class DrawBoxTool extends AbstractTool {
        entity: BoxLabelEntity;
        _start_point: Vector2;
        _current_point: Vector2;

        constructor(view: RootLabelView, entity: BoxLabelEntity) {
            super(view);
            this.entity = entity;
            this._start_point = null;
            this._current_point = null;
        }

        on_init() {
        };

        on_shutdown() {
        };

        on_switch_in(pos: Vector2) {
            if (this._start_point !== null) {
                this._current_point = pos;
                this.update_box();
            }
        };

        on_switch_out(pos: Vector2) {
            if (this._start_point !== null) {
                this._current_point = null;
                this.update_box();
            }
        };

        on_cancel(pos: Vector2): boolean {
            if (this.entity !== null) {
                if (this._start_point !== null) {
                    this.destroy_entity();
                    this._start_point = null;
                }
            }
            else {
                this._view.unselect_all_entities();
                this._view.view.set_current_tool(new SelectEntityTool(this._view));
            }
            return true;
        };

        on_left_click(pos: Vector2, event: any) {
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

        on_move(pos: Vector2) {
            if (this._start_point !== null) {
                this._current_point = pos;
                this.update_box();
            }
        };



        create_entity(pos: Vector2) {
            var model = new_BoxLabelModel(pos, {x: 0.0, y: 0.0});
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

        update_box() {
            if (this.entity !== null) {
                var box: AABox = null;
                if (this._start_point !== null) {
                    if (this._current_point !== null) {
                        box = AABox_from_points([this._start_point, this._current_point]);
                    }
                    else {
                        box = new AABox(this._start_point, this._start_point);
                    }
                }
                this.entity.model.centre = box.centre();
                this.entity.model.size = box.size();
                this.entity.update();
            }
        };
    }
}
