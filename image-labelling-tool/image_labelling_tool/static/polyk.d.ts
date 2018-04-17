declare module PolyK {
    export interface AABB {
        x: number;
        y: number;
        width: number;
        height: number;
    }

    export interface Intersection {
        dist: number;
        edge: number;
        norm: any;
        refl: any;
    }

    export function IsSimple(p: number[]): boolean;
    export function IsConvex(p: number[]): boolean;
    export function GetArea(p: number[]): number;
    export function GetAABB(p: number[]): AABB;
    export function Reverse(p: number[]): number[];
    export function Triangulate(p: number[]): number[];
    export function ContainsPoint(p: number[], px: number, py: number): boolean;
    export function Slice(p: number[], ax: number, ay: number, bx: number, by: number): number[];
    export function Raycast(p: number[], x: number, y: number, dx: number, dy: number, isc?: Intersection): Intersection;
    export function ClosestEdge(p: number[], x: number, y: number, isc?: Intersection): Intersection;
}