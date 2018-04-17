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
/// <reference path="./object_id_table.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./composite_label.ts" />
/// <reference path="./group_label.ts" />
/// <reference path="./main_tool.ts" />

module labelling_tool {
    export interface RootLabelViewListener {
        // Selection changed; update class selector dropdown
        on_selection_changed: (root_view: RootLabelView) => void;
        // Root list changed; queue push
        root_list_changed: (root_view: RootLabelView) => void;
    }

    /*
    Label view root
     */
    export class RootLabelView implements ContainerEntity {
        model: LabelHeaderModel;
        private _all_entities: AbstractLabelEntity<AbstractLabelModel>[];
        private root_entities: AbstractLabelEntity<AbstractLabelModel>[];
        private selected_entities: AbstractLabelEntity<AbstractLabelModel>[];
        private _label_model_obj_table: ObjectIDTable;
        private _label_model_id_to_entity: any;

        private root_listener: RootLabelViewListener;
        private _entity_event_listener: LabelEntityEventListener;

        view: LabellingTool;

        world: d3.Selection<any>;

        constructor(model: LabelHeaderModel, root_listener: RootLabelViewListener,
                    entity_listener: LabelEntityEventListener, ltool: LabellingTool,
                    world: d3.Selection<any>) {
            this.model = model;

            this._all_entities = [];
            this.root_entities = [];
            this.selected_entities = [];

            // Label model object table
            this._label_model_obj_table = new ObjectIDTable();
            // Label model object ID to entity
            this._label_model_id_to_entity = {};

            this.root_listener = root_listener;
            this._entity_event_listener = entity_listener;
            this.view = ltool;

            this.world = world;
        }


        /*
        Set model
         */
        set_model(model: LabelHeaderModel) {
            // Remove all entities
            var entites_to_shutdown = this.root_entities.slice();
            for (var i = 0; i < entites_to_shutdown.length; i++) {
                this.shutdown_entity(entites_to_shutdown[i]);
            }

            // Update the labels
            this.model = model;
            var labels = get_label_header_labels(this.model);

            // Set up the ID counter; ensure that it's value is 1 above the maximum label ID in use
            this._label_model_obj_table = new ObjectIDTable();
            this._label_model_obj_table.register_objects(labels);
            this._label_model_id_to_entity = {};

            // Reset the entity lists
            this._all_entities = [];
            this.root_entities = [];
            this.selected_entities = [];

            for (var i = 0; i < labels.length; i++) {
                var label = labels[i];
                var entity = this.get_or_create_entity_for_model(label);
                this.register_child(entity);
            }
        }

        /*
        Set complete
         */
        set_complete(complete: boolean) {
            this.model.complete = complete;
        }

        get_current_image_id(): string {
            if (this.model !== null  &&  this.model !== undefined) {
                return this.model.image_id;
            }
            else {
                return null;
            }
        };

        /*
        Set label visibility
         */
        set_label_visibility(visibility: LabelVisibility) {
            for (var i = 0; i < this._all_entities.length; i++) {
                this._all_entities[i].notify_hide_labels_change();
            }
        }


        /*
        Select an entity
         */
        select_entity(entity: AbstractLabelEntity<any>, multi_select: boolean, invert: boolean) {
            multi_select = multi_select === undefined  ?  false  :  multi_select;

            if (multi_select) {
                var index = this.selected_entities.indexOf(entity);
                var changed = false;

                if (invert) {
                    if (index === -1) {
                        // Add
                        this.selected_entities.push(entity);
                        entity.select(true);
                        changed = true;
                    }
                    else {
                        // Remove
                        this.selected_entities.splice(index, 1);
                        entity.select(false);
                        changed = true;
                    }
                }
                else {
                    if (index === -1) {
                        // Add
                        this.selected_entities.push(entity);
                        entity.select(true);
                        changed = true;
                    }
                }

                if (changed) {
                    this.root_listener.on_selection_changed(this);
                }
            }
            else {
                var prev_entity = this.get_selected_entity();

                if (prev_entity !== entity) {
                    for (var i = 0; i < this.selected_entities.length; i++) {
                        this.selected_entities[i].select(false);
                    }
                    this.selected_entities = [entity];
                    entity.select(true);
                }

                this.root_listener.on_selection_changed(this);
            }
        };


        /*
        Unselect all entities
         */
        unselect_all_entities() {
            for (var i = 0; i < this.selected_entities.length; i++) {
                this.selected_entities[i].select(false);
            }
            this.selected_entities = [];
            this.root_listener.on_selection_changed(this);
        };


        /*
        Get uniquely selected entity
         */
        get_selected_entity(): AbstractLabelEntity<AbstractLabelModel> {
            return this.selected_entities.length == 1  ?  this.selected_entities[0]  :  null;
        };

        /*
        Get selected entities
         */
        get_selection() {
            return this.selected_entities;
        };

        /*
        Get all entities
         */
        get_entities() {
            return this.root_entities;
        };



        /*
        Commit model
        invoke when a model is modified
        inserts the model into the tool data model and ensures that the relevant change events get send over
         */
        commit_model(model: AbstractLabelModel) {
            var labels = get_label_header_labels(this.model);
            var index = labels.indexOf(model);

            if (index !== -1) {
                this.root_listener.root_list_changed(this);
            }
        };


        /*
        Create composite label
         */
        create_composite_label_from_selection(): CompositeLabelEntity {
            var N = this.selected_entities.length;

            if (N > 0) {
                var model = new_CompositeLabelModel();

                for (var i = 0; i < this.selected_entities.length; i++) {
                    var model_id = ObjectIDTable.get_id(this.selected_entities[i].model);
                    model.components.push(model_id);
                }

                var entity = this.get_or_create_entity_for_model(model);
                this.add_child(entity);
                return entity;
            }
            else {
                return null;
            }
        }

        /*
        Create group label
         */
        create_group_label_from_selection(): GroupLabelEntity {
            var selection = this.selected_entities.slice();
            var N = selection.length;

            if (N > 0) {
                var model = new_GroupLabelModel();
                for (var i = 0; i < selection.length; i++) {
                    var entity = selection[i];
                    model.component_models.push(entity.model);
                    this.remove_child(entity);
                }

                var group_entity = this.get_or_create_entity_for_model(model);
                this.add_child(group_entity);
                return group_entity;
            }
            else {
                return null;
            }
        }

        /*
        Destroy selection
         */
        delete_selection(delete_filter_fn: (entity: AbstractLabelEntity<AbstractLabelModel>) => boolean) {
            var entities_to_remove: AbstractLabelEntity<AbstractLabelModel>[] = this.selected_entities.slice();
            var can_delete: boolean;

            this.unselect_all_entities();

            for (var i = 0; i < entities_to_remove.length; i++) {
                if (delete_filter_fn !== undefined && delete_filter_fn !== null) {
                    if (delete_filter_fn(entities_to_remove[i])) {
                        entities_to_remove[i].destroy();
                    }
                }
                else {
                    entities_to_remove[i].destroy();
                }
            }
        }


        /*
        Register and unregister entities
         */
        _register_entity(entity: AbstractLabelEntity<any>) {
            this._all_entities.push(entity);
            this._label_model_obj_table.register(entity.model);
            this._label_model_id_to_entity[entity.model.object_id] = entity;
        };

        _unregister_entity(entity: AbstractLabelEntity<any>) {
            var index = this._all_entities.indexOf(entity);

            if (index === -1) {
                throw "Attempting to unregister entity that is not in _all_entities";
            }

            // Notify all entities of the destruction of this model
            for (var i = 0; i < this._all_entities.length; i++) {
                if (i !== index) {
                    this._all_entities[i].notify_model_destroyed(entity.model);
                }
            }

            // Unregister in the ID to object table
            this._label_model_obj_table.unregister(entity.model);
            delete this._label_model_id_to_entity[entity.model.object_id];

            // Remove
            this._all_entities.splice(index, 1);
        };


        /*
        Initialise and shutdown entities
         */
        initialise_entity(entity: AbstractLabelEntity<AbstractLabelModel>) {
            entity.attach();
        };

        shutdown_entity(entity: AbstractLabelEntity<AbstractLabelModel>) {
            entity.detach();
        };


        /*
        Get entity for model ID
         */
        get_entity_for_model_id(model_id: number) {
            return this._label_model_id_to_entity[model_id];
        };

        /*
        Get entity for model
         */
        get_entity_for_model(model: AbstractLabelModel) {
            var model_id = ObjectIDTable.get_id(model);
            return this._label_model_id_to_entity[model_id];
        };

        /*
        Get or create entity for model
         */
        get_or_create_entity_for_model(model: AbstractLabelModel) {
            var model_id = ObjectIDTable.get_id(model);
            if (model_id === null ||
                !this._label_model_id_to_entity.hasOwnProperty(model_id)) {
                var entity = new_entity_for_model(this, model);
                this.initialise_entity(entity);
                return entity;
            }
            else {
                return this._label_model_id_to_entity[model_id];
            }
        };




        /*
        Register and unregister child entities
         */
        private register_child(entity: AbstractLabelEntity<any>) {
            this.root_entities.push(entity);
            entity.add_event_listener(this._entity_event_listener);
            entity.set_parent(this);
        };

        private unregister_child(entity: AbstractLabelEntity<any>) {
            // Remove from list of root entities
            var index_in_roots = this.root_entities.indexOf(entity);

            if (index_in_roots === -1) {
                throw "Attempting to unregister root entity that is not in root_entities";
            }

            this.root_entities.splice(index_in_roots, 1);

            // Remove from selection if present
            var index_in_selection = this.selected_entities.indexOf(entity);
            if (index_in_selection !== -1) {
                entity.select(false);
                this.selected_entities.splice(index_in_selection, 1);
            }

            entity.remove_event_listener(this._entity_event_listener);
            entity.set_parent(null);
        };



        /*
        Add entity:
        register the entity and add its label to the tool data model
         */
        add_child(child: AbstractLabelEntity<AbstractLabelModel>): void {
            this.register_child(child);

            var labels = get_label_header_labels(this.model);
            labels = labels.concat([child.model]);
            this.model = replace_label_header_labels(this.model, labels);

            this.root_listener.root_list_changed(this);
        };

        /*
        Remove entity
        unregister the entity and remove its label from the tool data model
         */
        remove_child(child: AbstractLabelEntity<AbstractLabelModel>): void {
            // Get the label model
            var labels = get_label_header_labels(this.model);
            var index = labels.indexOf(child.model);
            if (index === -1) {
                throw "Attempting to remove root label that is not present";
            }
            // Remove the model from the label model array
            labels = labels.slice(0, index).concat(labels.slice(index+1));
            // Replace the labels in the label header
            this.model = replace_label_header_labels(this.model, labels);

            this.unregister_child(child);

            // Commit changes
            this.root_listener.root_list_changed(this);
        };
    }
}
