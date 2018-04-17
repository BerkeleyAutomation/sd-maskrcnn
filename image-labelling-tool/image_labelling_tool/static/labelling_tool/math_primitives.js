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
var labelling_tool;
(function (labelling_tool) {
    function ensure_config_option_exists(x, flag_name, default_value) {
        var v = x[flag_name];
        if (v === undefined) {
            x[flag_name] = default_value;
        }
        return x[flag_name];
    }
    labelling_tool.ensure_config_option_exists = ensure_config_option_exists;
    function compute_centroid_of_points(vertices) {
        var sum = [0.0, 0.0];
        var N = vertices.length;
        if (N === 0) {
            return { x: 0, y: 0 };
        }
        else {
            for (var i = 0; i < N; i++) {
                var vtx = vertices[i];
                sum[0] += vtx.x;
                sum[1] += vtx.y;
            }
            var scale = 1.0 / N;
            return { x: sum[0] * scale, y: sum[1] * scale };
        }
    }
    labelling_tool.compute_centroid_of_points = compute_centroid_of_points;
    function compute_sqr_length(v) {
        return v.x * v.x + v.y * v.y;
    }
    labelling_tool.compute_sqr_length = compute_sqr_length;
    function compute_sqr_dist(a, b) {
        var dx = b.x - a.x, dy = b.y - a.y;
        return dx * dx + dy * dy;
    }
    labelling_tool.compute_sqr_dist = compute_sqr_dist;
    /*
    RGBA colour
     */
    var Colour4 = (function () {
        function Colour4(r, g, b, a) {
            this.r = r;
            this.g = g;
            this.b = b;
            this.a = a;
        }
        Colour4.prototype.lerp = function (x, t) {
            var s = 1.0 - t;
            return new Colour4(Math.round(this.r * s + x.r * t), Math.round(this.g * s + x.g * t), Math.round(this.b * s + x.b * t), this.a * s + x.a * t);
        };
        Colour4.prototype.lighten = function (amount) {
            return this.lerp(Colour4.WHITE, amount);
        };
        Colour4.prototype.with_alpha = function (alpha) {
            return new Colour4(this.r, this.g, this.b, alpha);
        };
        Colour4.prototype.to_rgba_string = function () {
            return 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + this.a + ')';
        };
        Colour4.from_rgb_a = function (rgb, alpha) {
            return new Colour4(rgb[0], rgb[1], rgb[2], alpha);
        };
        Colour4.BLACK = new Colour4(0, 0, 0, 1.0);
        Colour4.WHITE = new Colour4(255, 255, 255, 1.0);
        return Colour4;
    }());
    labelling_tool.Colour4 = Colour4;
    /*
    Axis-aligned box
     */
    var AABox = (function () {
        function AABox(lower, upper) {
            this.lower = lower;
            this.upper = upper;
        }
        AABox.prototype.contains_point = function (point) {
            return point.x >= this.lower.x && point.x <= this.upper.x &&
                point.y >= this.lower.y && point.y <= this.upper.y;
        };
        AABox.prototype.centre = function () {
            return { x: (this.lower.x + this.upper.x) * 0.5,
                y: (this.lower.y + this.upper.y) * 0.5 };
        };
        AABox.prototype.size = function () {
            return { x: this.upper.x - this.lower.x,
                y: this.upper.y - this.lower.y };
        };
        AABox.prototype.closest_point_to = function (p) {
            var x = Math.max(this.lower.x, Math.min(this.upper.x, p.x));
            var y = Math.max(this.lower.y, Math.min(this.upper.y, p.y));
            return { x: x, y: y };
        };
        AABox.prototype.sqr_distance_to = function (p) {
            var c = this.closest_point_to(p);
            var dx = c.x - p.x, dy = c.y - p.y;
            return dx * dx + dy * dy;
        };
        AABox.prototype.distance_to = function (p) {
            return Math.sqrt(this.sqr_distance_to(p));
        };
        return AABox;
    }());
    labelling_tool.AABox = AABox;
    function AABox_from_points(array_of_points) {
        if (array_of_points.length > 0) {
            var first = array_of_points[0];
            var lower = { x: first.x, y: first.y };
            var upper = { x: first.x, y: first.y };
            for (var i = 1; i < array_of_points.length; i++) {
                var p = array_of_points[i];
                lower.x = Math.min(lower.x, p.x);
                lower.y = Math.min(lower.y, p.y);
                upper.x = Math.max(upper.x, p.x);
                upper.y = Math.max(upper.y, p.y);
            }
            return new AABox(lower, upper);
        }
        else {
            return new AABox({ x: 0, y: 0 }, { x: 0, y: 0 });
        }
    }
    labelling_tool.AABox_from_points = AABox_from_points;
    function AABox_from_aaboxes(array_of_boxes) {
        if (array_of_boxes.length > 0) {
            var first = array_of_boxes[0];
            var result = new AABox({ x: first.lower.x, y: first.lower.y }, { x: first.upper.x, y: first.upper.y });
            for (var i = 1; i < array_of_boxes.length; i++) {
                var box = array_of_boxes[i];
                result.lower.x = Math.min(result.lower.x, box.lower.x);
                result.lower.y = Math.min(result.lower.y, box.lower.y);
                result.upper.x = Math.max(result.upper.x, box.upper.x);
                result.upper.y = Math.max(result.upper.y, box.upper.y);
            }
            return result;
        }
        else {
            return new AABox({ x: 1, y: 1 }, { x: -1, y: -1 });
        }
    }
    labelling_tool.AABox_from_aaboxes = AABox_from_aaboxes;
})(labelling_tool || (labelling_tool = {}));
