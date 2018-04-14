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
    Group label model
     */
    interface GroupLabelModel extends AbstractLabelModel {
        component_models: AbstractLabelModel[];
    }

    export function new_GroupLabelModel(): GroupLabelModel {
        return {label_type: 'group', label_class: null, component_models: []};
    }



    /*
    Group label entity
     */
    export class GroupLabelEntity extends AbstractLabelEntity<GroupLabelModel> implements ContainerEntity {
        _component_entities: AbstractLabelEntity<AbstractLabelModel>[];
        _bounding_rect: any;
        _bounding_aabox: AABox;
        _component_event_listener: LabelEntityEventListener;


        constructor(view: RootLabelView, model: GroupLabelModel) {
            super(view, model);
            var self = this;
            this._component_event_listener = {
                on_mouse_in: (entity) => {
                    for (var i = 0; i < self._event_listeners.length; i++) {
                        self._event_listeners[i].on_mouse_in(self);
                    }
                },
                on_mouse_out: (entity) => {
                    for (var i = 0; i < self._event_listeners.length; i++) {
                        self._event_listeners[i].on_mouse_out(self);
                    }
                }
            };
        }


        add_child(child: AbstractLabelEntity<AbstractLabelModel>): void {
            this.model.component_models.push(child.model);
            this._component_entities.push(child);
            child.add_event_listener(this._component_event_listener);
            child.set_parent(this);

            this.update_bbox();
            this.update();
            this._update_style();
        }

        remove_child(child: AbstractLabelEntity<AbstractLabelModel>): void {
            var index = this.model.component_models.indexOf(child.model);

            if (index === -1) {
                throw "GroupLabelEntity.remove_child: could not find child model";
            }

            this.model.component_models.splice(index, 1);
            this._component_entities.splice(index, 1);
            child.remove_event_listener(this._component_event_listener);
            child.set_parent(null);

            this.update_bbox();
            this.update();
            this._update_style();
        }

        remove_all_children(): void {
            for (var i = 0; i < this._component_entities.length; i++) {
                var child = this._component_entities[i];
                child.remove_event_listener(this._component_event_listener);
                child.set_parent(null);
            }

            this.model.component_models = [];
            this._component_entities = [];

            this.update_bbox();
            this.update();
            this._update_style();
        }


        attach() {
            super.attach();

            this._bounding_rect = this.root_view.world.append("rect")
                .attr("pointer-events", "none")
                .attr("x", 0).attr("y", 0)
                .attr("width", 0).attr("height", 0)
                .attr("visibility", "hidden");

            // Initialise child entities
            this._component_entities = [];
            var component_bboxes: AABox[] = [];
            for (var i = 0; i < this.model.component_models.length; i++) {
                var model = this.model.component_models[i];
                var model_entity = this.root_view.get_or_create_entity_for_model(model);
                this._component_entities.push(model_entity);
                component_bboxes.push(model_entity.compute_bounding_box());
                model_entity.add_event_listener(this._component_event_listener);
                model_entity.set_parent(this);
            }
            this._bounding_aabox = AABox_from_aaboxes(component_bboxes);

            this.update();
            this._update_style();
        };

        detach() {
            for (var i = 0; i < this._component_entities.length; i++) {
                var entity = this._component_entities[i];
                this.root_view.shutdown_entity(entity);
            }
            this._bounding_rect.remove();
            super.detach();
        };

        destroy() {
            var children = this._component_entities.slice();

            this.remove_all_children();

            for (var i = 0; i < children.length; i++) {
                this.parent_entity.add_child(children[i]);
            }

            this.parent_entity.remove_child(this);
            this.root_view.shutdown_entity(this);

            this._component_entities = [];
        }

        private update_bbox() {
            var component_bboxes: AABox[] = [];
            for (var i = 0; i < this._component_entities.length; i++) {
                var entity = this._component_entities[i];
                component_bboxes.push(entity.compute_bounding_box());
            }
            this._bounding_aabox = AABox_from_aaboxes(component_bboxes);
        }

        update() {
            var size = this._bounding_aabox.size();
            this._bounding_rect
                .attr('x', this._bounding_aabox.lower.x)
                .attr('y', this._bounding_aabox.lower.y)
                .attr('width', size.x)
                .attr('height', size.y);
        }

        commit() {
            this.root_view.commit_model(this.model);
        }




        select(state: boolean) {
            for (var i = 0; i < this._component_entities.length; i++) {
                this._component_entities[i].select(state);
            }
            super.select(state);
        }

        hover(state: boolean) {
            for (var i = 0; i < this._component_entities.length; i++) {
                this._component_entities[i].hover(state);
            }
            super.hover(state);
        }


        set_label_class(label_class: string) {
            for (var i = 0; i < this._component_entities.length; i++) {
                this._component_entities[i].set_label_class(label_class);
            }
            super.set_label_class(label_class);
        }


        _update_style() {
            if (this._attached) {
                if (this._selected) {
                    if (this._hover) {
                        this._bounding_rect.attr("style", "stroke:rgba(192,128,255,0.8); fill:rgba(192,128,255,0.2); line-width: 1.0px;")
                            .attr("visibility", "visible");
                    }
                    else {
                        this._bounding_rect.attr("style", "stroke:rgba(192,128,255,0.6); fill:none; line-width: 1.0px;")
                            .attr("visibility", "visible");
                    }
                }
                else {
                    if (this._hover) {
                        this._bounding_rect.attr("style", "stroke:rgba(192,128,255,0.4); fill:none; line-width: 1.0px;")
                            .attr("visibility", "visible");
                    }
                    else {
                        this._bounding_rect.attr("visibility", "hidden");
                    }
                }
            }
        }

        _compute_component_centroids(): Vector2[] {
            var component_centroids: Vector2[] = [];
            for (var i = 0; i < this._component_entities.length; i++) {
                var entity = this._component_entities[i];
                var centroid = entity.compute_centroid();
                component_centroids.push(centroid);
            }
            return component_centroids;
        };

        compute_centroid(): Vector2 {
            return this._bounding_aabox.centre();
        };

        compute_bounding_box(): AABox {
            return this._bounding_aabox;
        };

        contains_pointer_position(point: Vector2): boolean {
            if (this.compute_bounding_box().contains_point(point)) {
                for (var i = 0; i < this._component_entities.length; i++) {
                    if (this._component_entities[i].contains_pointer_position(point)) {
                        return true;
                    }
                }
                return false;
            }
            else {
                return false;
            }
        }

        distance_to_point(point: Vector2): number {
            var best_dist: number = null;
            for (var i = 0; i < this._component_entities.length; i++) {
                var entity = this._component_entities[i];
                var d = entity.distance_to_point(point);
                if (d !== null) {
                    if (best_dist === null || d < best_dist) {
                        best_dist = d;
                    }
                }
            }
            return best_dist;
        }
    }


    register_entity_factory('group', (root_view: RootLabelView, model: AbstractLabelModel) => {
        return new GroupLabelEntity(root_view, model as GroupLabelModel);
    });
}
