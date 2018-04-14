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
var labelling_tool;
(function (labelling_tool) {
    /*
    Select entity tool
     */
    var SelectEntityTool = (function (_super) {
        __extends(SelectEntityTool, _super);
        function SelectEntityTool(view) {
            var _this = _super.call(this, view) || this;
            var self = _this;
            _this._entities_under_pointer = [];
            _this._current_entity = null;
            _this._current_entity_index = null;
            _this._key_event_listener = function (event) {
                self.on_key_press(event);
            };
            return _this;
        }
        SelectEntityTool.prototype.on_init = function () {
            this._entities_under_pointer = [];
            this._current_entity = null;
        };
        ;
        SelectEntityTool.prototype.on_shutdown = function () {
            // Remove any hover
            if (this._current_entity !== null) {
                this._current_entity.hover(false);
            }
        };
        ;
        SelectEntityTool.prototype.on_switch_in = function (pos) {
            document.addEventListener("keypress", this._key_event_listener);
        };
        ;
        SelectEntityTool.prototype.on_switch_out = function (pos) {
            document.removeEventListener("keypress", this._key_event_listener);
        };
        ;
        SelectEntityTool.prototype.on_key_press = function (event) {
            var key = event.key;
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
        };
        SelectEntityTool.prototype.on_left_click = function (pos, event) {
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
        ;
        SelectEntityTool.prototype.on_move = function (pos) {
            this._update_entities_under_pointer(pos);
        };
        ;
        SelectEntityTool.prototype._update_entities_under_pointer = function (pos) {
            var prev_under = this._entities_under_pointer;
            var under = this._get_entities_under_pointer(pos);
            var changed = false;
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
                var prev = this._current_entity;
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
        };
        SelectEntityTool.prototype._get_entities_under_pointer = function (pos) {
            var entities = this._view.get_entities();
            var entities_under_pointer = [];
            for (var i = 0; i < entities.length; i++) {
                var entity = entities[i];
                if (entity.contains_pointer_position(pos)) {
                    entities_under_pointer.push(entity);
                }
            }
            return entities_under_pointer;
        };
        SelectEntityTool._current_entity_modified = function (prev, cur) {
            if (cur !== prev) {
                if (prev !== null) {
                    prev.hover(false);
                }
                if (cur !== null) {
                    cur.hover(true);
                }
            }
        };
        ;
        return SelectEntityTool;
    }(labelling_tool.AbstractTool));
    labelling_tool.SelectEntityTool = SelectEntityTool;
    /*
    Brush select entity tool
     */
    var BrushSelectEntityTool = (function (_super) {
        __extends(BrushSelectEntityTool, _super);
        function BrushSelectEntityTool(view) {
            var _this = _super.call(this, view) || this;
            _this._highlighted_entities = [];
            _this._brush_radius = 10.0;
            _this._brush_circle = null;
            return _this;
        }
        BrushSelectEntityTool.prototype.on_init = function () {
            this._highlighted_entities = [];
            this._brush_circle = this._view.world.append("circle");
            this._brush_circle.attr("r", this._brush_radius);
            this._brush_circle.attr("visibility", "hidden");
            this._brush_circle.style("fill", "rgba(128,0,0,0.05)");
            this._brush_circle.style("stroke-width", "1.0");
            this._brush_circle.style("stroke", "red");
        };
        ;
        BrushSelectEntityTool.prototype.on_shutdown = function () {
            this._brush_circle.remove();
            this._brush_circle = null;
            this._highlighted_entities = [];
        };
        ;
        BrushSelectEntityTool.prototype._get_entities_in_range = function (point) {
            var in_range = [];
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
        ;
        BrushSelectEntityTool.prototype._highlight_entities = function (entities) {
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
        ;
        BrushSelectEntityTool.prototype.on_button_down = function (pos, event) {
            this._highlight_entities([]);
            var entities = this._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                this._view.select_entity(entities[i], event.shiftKey || i > 0, false);
            }
            return true;
        };
        ;
        BrushSelectEntityTool.prototype.on_button_up = function (pos, event) {
            this._highlight_entities(this._get_entities_in_range(pos));
            return true;
        };
        ;
        BrushSelectEntityTool.prototype.on_move = function (pos) {
            this._highlight_entities(this._get_entities_in_range(pos));
            this._brush_circle.attr("cx", pos.x);
            this._brush_circle.attr("cy", pos.y);
        };
        ;
        BrushSelectEntityTool.prototype.on_drag = function (pos) {
            var entities = this._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                this._view.select_entity(entities[i], true, false);
            }
            this._brush_circle.attr("cx", pos.x);
            this._brush_circle.attr("cy", pos.y);
            return true;
        };
        ;
        BrushSelectEntityTool.prototype.on_wheel = function (pos, wheelDeltaX, wheelDeltaY) {
            this._brush_radius += wheelDeltaY * 0.1;
            this._brush_radius = Math.max(this._brush_radius, 1.0);
            this._brush_circle.attr("r", this._brush_radius);
            return true;
        };
        ;
        BrushSelectEntityTool.prototype.on_key_down = function (event) {
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
        ;
        BrushSelectEntityTool.prototype.on_switch_in = function (pos) {
            this._highlight_entities(this._get_entities_in_range(pos));
            this._brush_circle.attr("visibility", "visible");
        };
        ;
        BrushSelectEntityTool.prototype.on_switch_out = function (pos) {
            this._highlight_entities([]);
            this._brush_circle.attr("visibility", "hidden");
        };
        ;
        return BrushSelectEntityTool;
    }(labelling_tool.AbstractTool));
    labelling_tool.BrushSelectEntityTool = BrushSelectEntityTool;
})(labelling_tool || (labelling_tool = {}));
