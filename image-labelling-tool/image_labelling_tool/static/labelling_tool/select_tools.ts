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

module labelling_tool {
    /*
    Select entity tool
     */
    export class SelectEntityTool extends AbstractTool {
        _entities_under_pointer: AbstractLabelEntity<AbstractLabelModel>[];
        _current_entity: AbstractLabelEntity<AbstractLabelModel>;
        _current_entity_index: number;
        _key_event_listener: (event: any)=>any;

        constructor(view: RootLabelView) {
            super(view);
            var self = this;
            this._entities_under_pointer = [];
            this._current_entity = null;
            this._current_entity_index = null;
            this._key_event_listener = function(event) {
                self.on_key_press(event);
            }
        }

        on_init() {
            this._entities_under_pointer = [];
            this._current_entity = null;
        };

        on_shutdown() {
            // Remove any hover
            if (this._current_entity !== null) {
                this._current_entity.hover(false);
            }
        };

        on_switch_in(pos: Vector2) {
            document.addEventListener("keypress", this._key_event_listener);
        };

        on_switch_out(pos: Vector2) {
            document.removeEventListener("keypress", this._key_event_listener);
        };

        on_key_press(event: any) {
            var key: string = event.key;
            if (key === '[' || key === ']') {
                // 91: Open square bracket
                // 93: Close square bracket
                var prev = this._current_entity;

                if (key === '[') {
                    console.log('Backward through ' + this._entities_under_pointer.length);
                    this._current_entity_index--;
                    if (this._current_entity_index < 0) {
                        this._current_entity_index = this._entities_under_pointer.length - 1;
                    }
                }
                else if (key === ']') {
                    console.log('Forward through ' + this._entities_under_pointer.length);
                    this._current_entity_index++;
                    if (this._current_entity_index >= this._entities_under_pointer.length) {
                        this._current_entity_index = 0;
                    }
                }

                this._current_entity = this._entities_under_pointer[this._current_entity_index];
                SelectEntityTool._current_entity_modified(prev, this._current_entity);
            }
        }

        on_left_click(pos: Vector2, event: any) {
            this._update_entities_under_pointer(pos);
            if (this._current_entity !== null) {
                this._view.select_entity(this._current_entity, event.shiftKey, true);
            }
            else {
                if (!event.shiftKey) {
                    this._view.unselect_all_entities();
                }
            }
        };

        on_move(pos: Vector2) {
            this._update_entities_under_pointer(pos);
        };

        _update_entities_under_pointer(pos: Vector2) {
            var prev_under: AbstractLabelEntity<AbstractLabelModel>[] = this._entities_under_pointer;
            var under: AbstractLabelEntity<AbstractLabelModel>[] = this._get_entities_under_pointer(pos);

            var changed: boolean = false;
            if (prev_under.length == under.length) {
                for (var i = 0; i < prev_under.length; i++) {
                    if (prev_under[i].get_entity_id() !== under[i].get_entity_id()) {
                        changed = true;
                        break;
                    }
                }
            }
            else {
                changed = true;
            }

            if (changed) {
                var prev: AbstractLabelEntity<AbstractLabelModel> = this._current_entity;
                this._entities_under_pointer = under;
                if (this._entities_under_pointer.length > 0) {
                    this._current_entity_index = this._entities_under_pointer.length - 1;
                    this._current_entity = this._entities_under_pointer[this._current_entity_index];
                }
                else {
                    this._current_entity_index = null;
                    this._current_entity = null;
                }
                SelectEntityTool._current_entity_modified(prev, this._current_entity);
            }
        }

        _get_entities_under_pointer(pos: Vector2): AbstractLabelEntity<AbstractLabelModel>[] {
            var entities: AbstractLabelEntity<AbstractLabelModel>[] = this._view.get_entities();
            var entities_under_pointer: AbstractLabelEntity<AbstractLabelModel>[] = [];
            for (var i = 0; i < entities.length; i++) {
                var entity = entities[i];
                if (entity.contains_pointer_position(pos)) {
                    entities_under_pointer.push(entity);
                }
            }
            return entities_under_pointer;
        }

        static _current_entity_modified(prev: AbstractLabelEntity<AbstractLabelModel>, cur: AbstractLabelEntity<AbstractLabelModel>) {
            if (cur !== prev) {
                if (prev !== null) {
                    prev.hover(false);
                }

                if (cur !== null) {
                    cur.hover(true);
                }
            }
        };
    }


    /*
    Brush select entity tool
     */
    export class BrushSelectEntityTool extends AbstractTool {
        _highlighted_entities: AbstractLabelEntity<AbstractLabelModel>[];
        _brush_radius: number;
        _brush_circle: any;

        constructor(view: RootLabelView) {
            super(view);

            this._highlighted_entities = [];
            this._brush_radius = 10.0;
            this._brush_circle = null;
        }

        on_init() {
            this._highlighted_entities = [];
            this._brush_circle = this._view.world.append("circle");
            this._brush_circle.attr("r", this._brush_radius);
            this._brush_circle.attr("visibility", "hidden");
            this._brush_circle.style("fill", "rgba(128,0,0,0.05)");
            this._brush_circle.style("stroke-width", "1.0");
            this._brush_circle.style("stroke", "red");
        };

        on_shutdown() {
            this._brush_circle.remove();
            this._brush_circle = null;
            this._highlighted_entities = [];
        };


        _get_entities_in_range(point: Vector2) {
            var in_range: any[] = [];
            var entities = this._view.get_entities();
            for (var i = 0; i < entities.length; i++) {
                var entity = entities[i];
                var dist = entity.distance_to_point(point);
                if (dist !== null) {
                    if (dist <= this._brush_radius) {
                        in_range.push(entity);
                    }
                }
            }
            return in_range;
        };

        _highlight_entities(entities: AbstractLabelEntity<AbstractLabelModel>[]) {
            // Remove any hover
            for (var i = 0; i < this._highlighted_entities.length; i++) {
                this._highlighted_entities[i].hover(false);
            }

            this._highlighted_entities = entities;

            // Add hover
            for (var i = 0; i < this._highlighted_entities.length; i++) {
                this._highlighted_entities[i].hover(true);
            }
        };


        on_button_down(pos: Vector2, event: any) {
            this._highlight_entities([]);
            var entities = this._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                this._view.select_entity(entities[i], event.shiftKey || i > 0, false);
            }
            return true;
        };

        on_button_up(pos: Vector2, event: any) {
            this._highlight_entities(this._get_entities_in_range(pos));
            return true;
        };

        on_move(pos: Vector2) {
            this._highlight_entities(this._get_entities_in_range(pos));
            this._brush_circle.attr("cx", pos.x);
            this._brush_circle.attr("cy", pos.y);
        };

        on_drag(pos: Vector2): boolean {
            var entities = this._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                this._view.select_entity(entities[i], true, false);
            }
            this._brush_circle.attr("cx", pos.x);
            this._brush_circle.attr("cy", pos.y);
            return true;
        };

        on_wheel(pos: Vector2, wheelDeltaX: number, wheelDeltaY: number): boolean {
            this._brush_radius += wheelDeltaY * 0.1;
            this._brush_radius = Math.max(this._brush_radius, 1.0);
            this._brush_circle.attr("r", this._brush_radius);
            return true;
        };

        on_key_down(event: any): boolean {
            var handled = false;
            if (event.keyCode == 219) {
                this._brush_radius -= 2.0;
                handled = true;
            }
            else if (event.keyCode == 221) {
                this._brush_radius += 2.0;
                handled = true;
            }
            if (handled) {
                this._brush_radius = Math.max(this._brush_radius, 1.0);
                this._brush_circle.attr("r", this._brush_radius);
            }
            return handled;
        };

        on_switch_in(pos: Vector2) {
            this._highlight_entities(this._get_entities_in_range(pos));
            this._brush_circle.attr("visibility", "visible");
        };

        on_switch_out(pos: Vector2) {
            this._highlight_entities([]);
            this._brush_circle.attr("visibility", "hidden");
        };
    }
}
