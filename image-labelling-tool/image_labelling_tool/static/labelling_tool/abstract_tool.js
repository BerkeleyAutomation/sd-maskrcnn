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
var labelling_tool;
(function (labelling_tool) {
    /*
    Abstract tool
     */
    var AbstractTool = (function () {
        function AbstractTool(view) {
            this._view = view;
        }
        AbstractTool.prototype.on_init = function () {
        };
        ;
        AbstractTool.prototype.on_shutdown = function () {
        };
        ;
        AbstractTool.prototype.on_switch_in = function (pos) {
        };
        ;
        AbstractTool.prototype.on_switch_out = function (pos) {
        };
        ;
        AbstractTool.prototype.on_left_click = function (pos, event) {
        };
        ;
        AbstractTool.prototype.on_cancel = function (pos) {
            return false;
        };
        ;
        AbstractTool.prototype.on_button_down = function (pos, event) {
        };
        ;
        AbstractTool.prototype.on_button_up = function (pos, event) {
        };
        ;
        AbstractTool.prototype.on_move = function (pos) {
        };
        ;
        AbstractTool.prototype.on_drag = function (pos) {
            return false;
        };
        ;
        AbstractTool.prototype.on_wheel = function (pos, wheelDeltaX, wheelDeltaY) {
            return false;
        };
        ;
        AbstractTool.prototype.on_key_down = function (event) {
            return false;
        };
        ;
        AbstractTool.prototype.on_entity_mouse_in = function (entity) {
        };
        ;
        AbstractTool.prototype.on_entity_mouse_out = function (entity) {
        };
        ;
        return AbstractTool;
    }());
    labelling_tool.AbstractTool = AbstractTool;
})(labelling_tool || (labelling_tool = {}));
