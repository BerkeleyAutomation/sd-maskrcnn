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

module labelling_tool {
    /*
    Abstract tool
     */
    export class AbstractTool{
        _view: RootLabelView;

        constructor(view: RootLabelView) {
            this._view = view;

        }

        on_init() {
        };

        on_shutdown() {
        };

        on_switch_in(pos: Vector2) {
        };

        on_switch_out(pos: Vector2) {
        };

        on_left_click(pos: Vector2, event: any) {
        };

        on_cancel(pos: Vector2): boolean {
            return false;
        };

        on_button_down(pos: Vector2, event: any) {
        };

        on_button_up(pos: Vector2, event: any) {
        };

        on_move(pos: Vector2) {
        };

        on_drag(pos: Vector2): boolean {
            return false;
        };

        on_wheel(pos: Vector2, wheelDeltaX: number, wheelDeltaY: number): boolean {
            return false;
        };

        on_key_down(event: any): boolean {
            return false;
        };

        on_entity_mouse_in(entity: AbstractLabelEntity<AbstractLabelModel>) {
        };

        on_entity_mouse_out(entity: AbstractLabelEntity<AbstractLabelModel>) {
        };
    }
}
