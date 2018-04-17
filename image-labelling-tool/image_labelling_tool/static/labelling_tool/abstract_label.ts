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

/// <reference path="../d3.d.ts" />
/// <reference path="./math_primitives.ts" />
/// <reference path="./root_label_view.ts" />

module labelling_tool {
    /*
    Abstract label model
     */
    export interface AbstractLabelModel {
        label_type: string;
        label_class: string;
    }



    /*
    Label visibility
     */
    export enum LabelVisibility {
        HIDDEN,
        FAINT,
        FULL
    }


    export interface LabelEntityEventListener {
        on_mouse_in: (entity: any) => void;
        on_mouse_out: (entity: any) => void;
    }


    /*
    Abstract label entity
     */
    export class AbstractLabelEntity<ModelType extends AbstractLabelModel> {
        private static entity_id_counter: number = 0;
        model: ModelType;
        protected root_view: RootLabelView;
        private entity_id: number;
        _attached: boolean;
        _hover: boolean;
        _selected: boolean;
        _event_listeners: LabelEntityEventListener[];
        parent_entity: ContainerEntity;



        constructor(view: RootLabelView, model: ModelType) {
            this.root_view = view;
            this.model = model;
            this._attached = this._hover = this._selected = false;
            this._event_listeners = [];
            this.parent_entity = null;
            this.entity_id = AbstractLabelEntity.entity_id_counter++;

        }


        add_event_listener(listener: LabelEntityEventListener) {
            this._event_listeners.push(listener)
        }

        remove_event_listener(listener: LabelEntityEventListener) {
            var i = this._event_listeners.indexOf(listener);
            if (i !== -1) {
                this._event_listeners.splice(i, 1);
            }
        }

        set_parent(parent: ContainerEntity) {
            this.parent_entity = parent;
        }

        get_entity_id(): number {
            return this.entity_id;
        }

        attach() {
            this.root_view._register_entity(this);
            this._attached = true;
        }

        detach() {
            this._attached = false;
            this.root_view._unregister_entity(this);
        }

        destroy() {
            if (this.parent_entity !== null) {
                this.parent_entity.remove_child(this);
            }
            this.root_view.shutdown_entity(this);
        }

        update() {
        }

        commit() {
        }

        hover(state: boolean) {
            this._hover = state;
            this._update_style();
        }

        select(state: boolean) {
            this._selected = state;
            this._update_style();
        }

        notify_hide_labels_change() {
            this._update_style();
        }

        get_label_type_name(): string {
            return this.model.label_type;
        }

        get_label_class(): string {
            return this.model.label_class;
        }

        set_label_class(label_class: string) {
            this.model.label_class = label_class;
            this._update_style();
            this.commit();
        }

        _update_style() {
        };

        _outline_colour(): Colour4 {
            if (this._selected) {
                if (this._hover) {
                    return new Colour4(255, 0, 128, 1.0);
                }
                else {
                    return new Colour4(255, 0, 0, 1.0);
                }
            }
            else {
                if (this._hover) {
                    return new Colour4(0, 255, 128, 1.0);
                }
                else {
                    return new Colour4(255, 255, 0, 1.0);
                }
            }
        }

        compute_centroid(): Vector2 {
            return null;
        }

        compute_bounding_box(): AABox {
            return null;
        };

        contains_pointer_position(point: Vector2): boolean {
            return false;
        }

        distance_to_point(point: Vector2): number {
            return null;
        };

        notify_model_destroyed(model_id: number) {
        };
    }


        /*
    Container entity
     */
    export interface ContainerEntity {
        add_child(child: AbstractLabelEntity<AbstractLabelModel>): void;
        remove_child(child: AbstractLabelEntity<AbstractLabelModel>): void;
    }



    /*
    Map label type to entity constructor
     */
    var label_type_to_entity_factory: any = {};


    /*
    Register label entity factory
     */
    export function register_entity_factory(label_type_name: string,
                                            factory: (root_view:RootLabelView, model:AbstractLabelModel) => any) {
        label_type_to_entity_factory[label_type_name] = factory;
    }

    /*
    Construct entity for given label model.
    Uses the map above to choose the appropriate constructor
     */
    export function new_entity_for_model(root_view: RootLabelView, label_model: AbstractLabelModel) {
        var factory = label_type_to_entity_factory[label_model.label_type];
        return factory(root_view, label_model);
    }
}
