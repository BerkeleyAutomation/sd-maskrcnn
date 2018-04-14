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

module labelling_tool {
    /*
    2D Vector
     */
    export interface Vector2 {
        x: number;
        y: number;
    }

    export function ensure_config_option_exists(x: any, flag_name: string, default_value: any) {
        var v = x[flag_name];
        if (v === undefined) {
            x[flag_name] = default_value;
        }
        return x[flag_name];
    }

    export function compute_centroid_of_points(vertices: Vector2[]): Vector2 {
        var sum = [0.0, 0.0];
        var N = vertices.length;
        if (N === 0) {
            return {x: 0, y: 0};
        }
        else {
            for (var i = 0; i < N; i++) {
                var vtx = vertices[i];
                sum[0] += vtx.x;
                sum[1] += vtx.y;
            }
            var scale = 1.0 / N;
            return {x: sum[0] * scale, y: sum[1] * scale};
        }
    }

    export function compute_sqr_length(v: Vector2) {
        return v.x * v.x + v.y * v.y;
    }

    export function compute_sqr_dist(a: Vector2, b: Vector2) {
        var dx = b.x - a.x, dy = b.y - a.y;
        return dx * dx + dy * dy;
    }


    /*
    RGBA colour
     */
    export class Colour4 {
        r: number;
        g: number;
        b: number;
        a: number;

        constructor(r: number, g: number, b: number, a: number) {
            this.r = r;
            this.g = g;
            this.b = b;
            this.a = a;
        }

        lerp(x: Colour4, t: number) {
            let s = 1.0 - t;
            return new Colour4(Math.round(this.r*s + x.r*t),
                               Math.round(this.g*s + x.g*t),
                               Math.round(this.b*s + x.b*t),
                               this.a*s + x.a*t);
        }

        lighten(amount: number): Colour4 {
            return this.lerp(Colour4.WHITE, amount);
        }

        with_alpha(alpha: number): Colour4 {
            return new Colour4(this.r, this.g, this.b, alpha);
        }

        to_rgba_string(): string {
            return 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + this.a + ')';
        }

        static from_rgb_a(rgb: number[], alpha: number): Colour4 {
            return new Colour4(rgb[0], rgb[1], rgb[2], alpha);
        }

        static BLACK: Colour4 = new Colour4(0, 0, 0, 1.0);
        static WHITE: Colour4 = new Colour4(255, 255, 255, 1.0);
    }



    /*
    Axis-aligned box
     */
    export class AABox {
        lower: Vector2;
        upper: Vector2;

        constructor(lower: Vector2, upper: Vector2) {
            this.lower = lower;
            this.upper = upper;
        }

        contains_point(point: Vector2): boolean {
            return point.x >= this.lower.x && point.x <= this.upper.x &&
                   point.y >= this.lower.y && point.y <= this.upper.y;
        }

        centre(): Vector2 {
            return {x: (this.lower.x + this.upper.x) * 0.5,
                    y: (this.lower.y + this.upper.y) * 0.5};
        }

        size(): Vector2 {
            return {x: this.upper.x - this.lower.x,
                    y: this.upper.y - this.lower.y};
        }

        closest_point_to(p: Vector2): Vector2 {
            var x = Math.max(this.lower.x, Math.min(this.upper.x, p.x));
            var y = Math.max(this.lower.y, Math.min(this.upper.y, p.y));
            return {x: x, y: y};
        }

        sqr_distance_to(p: Vector2): number {
            var c = this.closest_point_to(p);
            var dx: number = c.x - p.x, dy: number = c.y - p.y;
            return dx * dx + dy * dy;
        }

        distance_to(p: Vector2): number {
            return Math.sqrt(this.sqr_distance_to(p));
        }
    }

    export function AABox_from_points(array_of_points: Vector2[]): AABox {
        if (array_of_points.length > 0) {
            var first = array_of_points[0];
            var lower = {x: first.x, y: first.y};
            var upper = {x: first.x, y: first.y};
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
            return new AABox({x: 0, y: 0}, {x: 0, y: 0});
        }
    }

    export function AABox_from_aaboxes(array_of_boxes: AABox[]): AABox {
        if (array_of_boxes.length > 0) {
            var first = array_of_boxes[0];
            var result = new AABox({x: first.lower.x, y: first.lower.y},
                                   {x: first.upper.x, y: first.upper.y});
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
            return new AABox({x: 1, y: 1}, {x: -1, y: -1});
        }
    }
}
