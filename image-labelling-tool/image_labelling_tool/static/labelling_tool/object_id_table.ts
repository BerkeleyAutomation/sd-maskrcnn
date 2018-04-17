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
    Object ID table
     */
    export class ObjectIDTable {
        _id_counter:number;
        _id_to_object: any;

        constructor() {
            this._id_counter = 1;
            this._id_to_object = {};
        }

        get(id:number):any {
            return this._id_to_object[id];
        }

        register(obj:any):void {
            var id:number;
            if ('object_id' in obj && obj.object_id !== null) {
                id = obj.object_id;
                this._id_counter = Math.max(this._id_counter, id + 1);
                this._id_to_object[id] = obj;
            }
            else {
                id = this._id_counter;
                this._id_counter += 1;
                this._id_to_object[id] = obj;
                obj.object_id = id;
            }
        }

        unregister(obj:any) {
            delete this._id_to_object[obj.object_id];
            obj.object_id = null;
        }


        register_objects(object_array:any[]) {
            var obj:any, id:number, i:number;

            for (i = 0; i < object_array.length; i++) {
                obj = object_array[i];
                if ('object_id' in obj && obj.object_id !== null) {
                    id = obj.object_id;
                    this._id_counter = Math.max(this._id_counter, id + 1);
                    this._id_to_object[id] = obj;
                }
            }

            for (i = 0; i < object_array.length; i++) {
                obj = object_array[i];

                if ('object_id' in obj && obj.object_id !== null) {

                }
                else {
                    id = this._id_counter;
                    this._id_counter += 1;
                    this._id_to_object[id] = obj;
                    obj.object_id = id;
                }
            }
        }

        static get_id(x: any) {
            if ('object_id' in x && x.object_id !== null) {
                return x.object_id;
            }
            else {
                return null;
            }
        }
    }
}
