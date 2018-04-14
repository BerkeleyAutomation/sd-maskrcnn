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
/// <reference path="../jquery.d.ts" />
/// <reference path="../polyk.d.ts" />
/// <reference path="./math_primitives.ts" />
/// <reference path="./object_id_table.ts" />
/// <reference path="./label_class.ts" />
/// <reference path="./abstract_label.ts" />
/// <reference path="./abstract_tool.ts" />
/// <reference path="./select_tools.ts" />
/// <reference path="./point_label.ts" />
/// <reference path="./box_label.ts" />
/// <reference path="./polygonal_label.ts" />
/// <reference path="./composite_label.ts" />
/// <reference path="./group_label.ts" />
var labelling_tool;
(function (labelling_tool) {
    labelling_tool.get_label_header_labels = function (label_header) {
        var labels = label_header.labels;
        if (labels === undefined || labels === null) {
            return [];
        }
        else {
            return labels;
        }
    };
    labelling_tool.replace_label_header_labels = function (label_header, labels) {
        return { image_id: label_header.image_id,
            complete: label_header.complete,
            timeElapsed: label_header.timeElapsed,
            state: label_header.state,
            labels: labels };
    };
    /*
   Labelling tool view; links to the server side data structures
    */
    var LabellingTool = (function () {
        function LabellingTool(element, label_classes, tool_width, tool_height, images, initial_image_index, requestLabelsCallback, sendLabelHeaderFn, getNextUnlockedImageIDCallback, config) {
            var _this = this;
            var self = this;
            if (LabellingTool._global_key_handler === undefined ||
                LabellingTool._global_key_handler_connected === undefined) {
                LabellingTool._global_key_handler = null;
                LabellingTool._global_key_handler_connected = false;
            }
            config = config || {};
            this._config = config;
            config.tools = config.tools || {};
            labelling_tool.ensure_config_option_exists(config.tools, 'imageSelector', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'labelClassSelector', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'brushSelect', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'drawPointLabel', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'drawBoxLabel', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'drawPolyLabel', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'compositeLabel', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'groupLabel', true);
            labelling_tool.ensure_config_option_exists(config.tools, 'deleteLabel', true);
            config.settings = config.settings || {};
            labelling_tool.ensure_config_option_exists(config.settings, 'inactivityTimeoutMS', 10000);
            config.tools.deleteConfig = config.tools.deleteConfig || {};
            config.tools.deleteConfig.typePermissions = config.tools.deleteConfig.typePermissions || {};
            labelling_tool.ensure_config_option_exists(config.tools.deleteConfig.typePermissions, 'point', true);
            labelling_tool.ensure_config_option_exists(config.tools.deleteConfig.typePermissions, 'box', true);
            labelling_tool.ensure_config_option_exists(config.tools.deleteConfig.typePermissions, 'polygon', true);
            labelling_tool.ensure_config_option_exists(config.tools.deleteConfig.typePermissions, 'composite', true);
            labelling_tool.ensure_config_option_exists(config.tools.deleteConfig.typePermissions, 'group', true);
            /*
            Entity event listener
             */
            this._entity_event_listener = {
                on_mouse_in: function (entity) {
                    if (_this._current_tool !== null) {
                        _this._current_tool.on_entity_mouse_in(entity);
                    }
                },
                on_mouse_out: function (entity) {
                    if (_this._current_tool !== null) {
                        _this._current_tool.on_entity_mouse_out(entity);
                    }
                }
            };
            /*
            Root view listener
             */
            this.root_view_listener = {
                // Selection changed; update class selector dropdown
                on_selection_changed: function (root_view) {
                    _this._update_label_class_menu_from_views(root_view.get_selection());
                },
                // Root list changed; queue push
                root_list_changed: function (root_view) {
                    _this._commitStopwatch();
                    _this.queue_push_label_data();
                }
            };
            // Model
            var initial_model = {
                image_id: '',
                complete: false,
                timeElapsed: 0.0,
                state: 'editable',
                labels: []
            };
            // Active tool
            this._current_tool = null;
            // Classes
            this.label_classes = [];
            for (var i = 0; i < label_classes.length; i++) {
                this.label_classes.push(new labelling_tool.LabelClass(label_classes[i]));
            }
            // Hide labels
            this.label_visibility = labelling_tool.LabelVisibility.FULL;
            // Button state
            this._button_down = false;
            // Labelling tool dimensions
            this._tool_width = tool_width;
            this._tool_height = tool_height;
            // List of Image descriptors
            this._images = images;
            // Number of images in dataset
            this._num_images = images.length;
            // Image dimensions
            this._image_width = 0;
            this._image_height = 0;
            // Loaded flags
            this._image_loaded = false;
            this._labels_loaded = false;
            this._image_initialised = false;
            // Stopwatch
            this._stopwatchStart = null;
            this._stopwatchCurrent = null;
            this._stopwatchHandle = null;
            // Data request callback; labelling tool will call this when it needs a new image to show
            this._requestLabelsCallback = requestLabelsCallback;
            // Send data callback; labelling tool will call this when it wants to commit data to the backend in response
            // to user action
            this._sendLabelHeaderFn = sendLabelHeaderFn;
            // Get unlocked image IDs callback: labelling tool will call this when the user wants to move to the
            // next available image that is not locked. If it is `null` or `undefined` then the button will not
            // be displayed to the user
            this._getNextUnlockedImageIDCallback = getNextUnlockedImageIDCallback;
            // Send data interval for storing interval ID for queued label send
            this._pushDataTimeout = null;
            // Frozen flag; while frozen, data will not be sent to backend
            this.frozen = false;
            var toolbar_width = 220;
            this._labelling_area_width = this._tool_width - toolbar_width;
            var labelling_area_x_pos = toolbar_width + 10;
            this._lockableControls = $();
            // A <div> element that surrounds the labelling tool
            var overall_border = $('<div style="border: 1px solid gray; width: ' + this._tool_width + 'px;"/>')
                .appendTo(element);
            var toolbar_container = $('<div style="position: relative;">').appendTo(overall_border);
            var toolbar = $('<div style="position: absolute; width: ' + toolbar_width +
                'px; padding: 4px; display: inline-block; background: #d0d0d0; border: 1px solid #a0a0a0; font-family: sans-serif;"/>').appendTo(toolbar_container);
            var labelling_area = $('<div style="width:' + this._labelling_area_width + 'px; margin-left: ' + labelling_area_x_pos + 'px"/>').appendTo(overall_border);
            /*
             *
             *
             * TOOLBAR CONTENTS
             *
             *
             */
            //
            // IMAGE SELECTOR
            //
            $('<p style="background: #b0b0b0;">Current image</p>').appendTo(toolbar);
            if (config.tools.imageSelector) {
                var _increment_image_index = function (offset) {
                    var image_id = self._get_current_image_id();
                    var index = self._image_id_to_index(image_id);
                    var new_index = index + offset;
                    new_index = Math.max(Math.min(new_index, self._images.length - 1), 0);
                    // Only trigger an image load if the index has changed and it is valid
                    if (new_index !== index && new_index < self._images.length) {
                        self.loadImage(self._images[new_index]);
                    }
                };
                var _next_unlocked_image = function () {
                    var image_id = self._get_current_image_id();
                    self._getNextUnlockedImageIDCallback(image_id);
                };
                this._image_index_input = $('<input type="text" style="width: 30px; vertical-align: middle;" name="image_index"/>').appendTo(toolbar);
                this._image_index_input.on('change', function () {
                    var index_str = self._image_index_input.val();
                    var index = parseInt(index_str) - 1;
                    index = Math.max(Math.min(index, self._images.length - 1), 0);
                    if (index < self._images.length) {
                        self.loadImage(self._images[index]);
                    }
                });
                $('<span>' + '/' + this._num_images + '</span>').appendTo(toolbar);
                $('<br/>').appendTo(toolbar);
                var prev_image_button = $('<button>Prev image</button>').appendTo(toolbar);
                prev_image_button.button({
                    text: false,
                    icons: { primary: "ui-icon-seek-prev" }
                }).click(function (event) {
                    _increment_image_index(-1);
                    event.preventDefault();
                });
                var next_image_button = $('<button>Next image</button>').appendTo(toolbar);
                next_image_button.button({
                    text: false,
                    icons: { primary: "ui-icon-seek-next" }
                }).click(function (event) {
                    _increment_image_index(1);
                    event.preventDefault();
                });
                if (this._getNextUnlockedImageIDCallback !== null && this._getNextUnlockedImageIDCallback !== undefined) {
                    var next_unlocked_image_button = $('<button>Next unlocked image</button>').appendTo(toolbar);
                    next_unlocked_image_button.button({
                        text: false,
                        icons: { primary: "ui-icon-unlocked" }
                    }).click(function (event) {
                        _next_unlocked_image();
                        event.preventDefault();
                    });
                }
            }
            this._lockNotification = $('<div style="display: none;"><p style="font-size: 0.75em; color: #c00000">' +
                'These labels are locked and cannot be edited. Someone else got there first :). ' +
                'Please choose another image (click the unlock button above to find the next unlocked image).</p></div>');
            this._lockNotification.appendTo(toolbar);
            $('<br/>').appendTo(toolbar);
            this._complete_checkbox = $('<input type="checkbox">Finished</input>').appendTo(toolbar);
            this._complete_checkbox.change(function (event, ui) {
                self.root_view.set_complete(event.target.checked);
                self.queue_push_label_data();
            });
            this._lockableControls = this._lockableControls.add(this._complete_checkbox);
            //
            // LABEL CLASS SELECTOR AND HIDE LABELS
            //
            $('<p style="background: #b0b0b0;">Labels</p>').appendTo(toolbar);
            if (config.tools.labelClassSelector) {
                this._label_class_selector_menu = $('<select name="label_class_selector"/>').appendTo(toolbar);
                for (var i = 0; i < this.label_classes.length; i++) {
                    var cls = this.label_classes[i];
                    $('<option value="' + cls.name + '">' + cls.human_name + '</option>').appendTo(this._label_class_selector_menu);
                }
                $('<option value="__unclassified" selected="false">UNCLASSIFIED</option>').appendTo(this._label_class_selector_menu);
                this._label_class_selector_menu.change(function (event, ui) {
                    var label_class_name = event.target.value;
                    if (label_class_name == '__unclassified') {
                        label_class_name = null;
                    }
                    var selection = self.root_view.get_selection();
                    for (var i = 0; i < selection.length; i++) {
                        selection[i].set_label_class(label_class_name);
                    }
                });
                this._lockableControls = this._lockableControls.add(this._label_class_selector_menu);
            }
            $('<br/><span>Label visibility:</span><br/>').appendTo(toolbar);
            this.label_vis_hidden_radio = $('<input type="radio" name="labelvis" value="hidden">hidden</input>').appendTo(toolbar);
            this.label_vis_faint_radio = $('<input type="radio" name="labelvis" value="faint">faint</input>').appendTo(toolbar);
            this.label_vis_full_radio = $('<input type="radio" name="labelvis" value="full" checked>full</input>').appendTo(toolbar);
            this.label_vis_hidden_radio.change(function (event, ui) {
                if (event.target.checked) {
                    self.set_label_visibility(labelling_tool.LabelVisibility.HIDDEN);
                }
            });
            this.label_vis_faint_radio.change(function (event, ui) {
                if (event.target.checked) {
                    self.set_label_visibility(labelling_tool.LabelVisibility.FAINT);
                }
            });
            this.label_vis_full_radio.change(function (event, ui) {
                if (event.target.checked) {
                    self.set_label_visibility(labelling_tool.LabelVisibility.FULL);
                }
            });
            //
            // Tool buttons:
            // Select, brush select, draw poly, composite, group, delete
            //
            $('<p style="background: #b0b0b0;">Tools</p>').appendTo(toolbar);
            var select_button = $('<button>Select</button>').appendTo(toolbar);
            select_button.button().click(function (event) {
                self.set_current_tool(new labelling_tool.SelectEntityTool(self.root_view));
                event.preventDefault();
            });
            this._lockableControls = this._lockableControls.add(select_button);
            if (config.tools.brushSelect) {
                var brush_select_button = $('<button>Brush select</button>').appendTo(toolbar);
                brush_select_button.button().click(function (event) {
                    self.set_current_tool(new labelling_tool.BrushSelectEntityTool(self.root_view));
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(brush_select_button);
            }
            if (config.tools.drawPointLabel) {
                var draw_point_button = $('<button>Add point</button>').appendTo(toolbar);
                draw_point_button.button().click(function (event) {
                    var current = self.root_view.get_selected_entity();
                    if (current instanceof labelling_tool.PointLabelEntity) {
                        self.set_current_tool(new labelling_tool.DrawPointTool(self.root_view, current));
                    }
                    else {
                        self.set_current_tool(new labelling_tool.DrawPointTool(self.root_view, null));
                    }
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(draw_point_button);
            }
            if (config.tools.drawBoxLabel) {
                var draw_box_button = $('<button>Draw box</button>').appendTo(toolbar);
                draw_box_button.button().click(function (event) {
                    var current = self.root_view.get_selected_entity();
                    if (current instanceof labelling_tool.BoxLabelEntity) {
                        self.set_current_tool(new labelling_tool.DrawBoxTool(self.root_view, current));
                    }
                    else {
                        self.set_current_tool(new labelling_tool.DrawBoxTool(self.root_view, null));
                    }
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(draw_box_button);
            }
            if (config.tools.drawPolyLabel) {
                var draw_polygon_button = $('<button>Draw poly</button>').appendTo(toolbar);
                draw_polygon_button.button().click(function (event) {
                    var current = self.root_view.get_selected_entity();
                    if (current instanceof labelling_tool.PolygonalLabelEntity) {
                        self.set_current_tool(new labelling_tool.DrawPolygonTool(self.root_view, current));
                    }
                    else {
                        self.set_current_tool(new labelling_tool.DrawPolygonTool(self.root_view, null));
                    }
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(draw_polygon_button);
            }
            if (config.tools.compositeLabel) {
                var composite_button = $('<button>Composite</button>').appendTo(toolbar);
                composite_button.button().click(function (event) {
                    self.root_view.create_composite_label_from_selection();
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(composite_button);
            }
            if (config.tools.groupLabel) {
                var group_button = $('<button>Group</button>').appendTo(toolbar);
                group_button.button().click(function (event) {
                    var group_entity = self.root_view.create_group_label_from_selection();
                    if (group_entity !== null) {
                        self.root_view.select_entity(group_entity, false, false);
                    }
                    event.preventDefault();
                });
                this._lockableControls = this._lockableControls.add(group_button);
            }
            var canDelete = function (entity) {
                var typeName = entity.get_label_type_name();
                var delPerm = config.tools.deleteConfig.typePermissions[typeName];
                if (delPerm === undefined) {
                    return true;
                }
                else {
                    return delPerm;
                }
            };
            if (config.tools.deleteLabel) {
                var delete_label_button = $('<button>Delete</button>').appendTo(toolbar);
                delete_label_button.button({
                    text: false,
                    icons: { primary: "ui-icon-trash" }
                }).click(function (event) {
                    if (!self._confirm_delete_visible) {
                        var cancel_button = $('<button>Cancel</button>').appendTo(self._confirm_delete);
                        var confirm_button = $('<button>Confirm delete</button>').appendTo(self._confirm_delete);
                        var remove_confirm_ui = function () {
                            cancel_button.remove();
                            confirm_button.remove();
                            self._confirm_delete_visible = false;
                        };
                        cancel_button.button().click(function (event) {
                            remove_confirm_ui();
                            event.preventDefault();
                        });
                        confirm_button.button().click(function (event) {
                            self.root_view.delete_selection(canDelete);
                            remove_confirm_ui();
                            event.preventDefault();
                        });
                        self._confirm_delete_visible = true;
                    }
                    event.preventDefault();
                });
                this._confirm_delete = $('<span/>').appendTo(toolbar);
                this._confirm_delete_visible = false;
                this._lockableControls = this._lockableControls.add(delete_label_button);
            }
            /*
             *
             * LABELLING AREA
             *
             */
            // Zoom callback
            function zoomed() {
                var zoom_event = d3.event;
                var t = zoom_event.translate, s = zoom_event.scale;
                self._zoom_xlat = t;
                self._zoom_scale = s;
                self._zoom_node.attr("transform", "translate(" + t[0] + "," + t[1] + ") scale(" + s + ")");
            }
            // Create d3.js panning and zooming behaviour
            var zoom_behaviour = d3.behavior.zoom()
                .on("zoom", zoomed);
            // Disable context menu so we can use right-click
            labelling_area[0].oncontextmenu = function () {
                return false;
            };
            // Create SVG element of the appropriate dimensions
            this._svg = d3.select(labelling_area[0])
                .append("svg:svg")
                .attr("width", this._labelling_area_width)
                .attr("height", this._tool_height)
                .call(zoom_behaviour);
            this._loading_notification = d3.select(labelling_area[0])
                .append("svg:svg")
                .attr("width", this._labelling_area_width)
                .attr("height", this._tool_height)
                .attr("style", "display: none");
            this._loading_notification.append("rect")
                .attr("x", "0px")
                .attr("y", "0px")
                .attr("width", "" + this._labelling_area_width + "px")
                .attr("height", "" + this._tool_height + "px")
                .attr("fill", "#404040");
            this._loading_notification_text = this._loading_notification.append("text")
                .attr("x", "50%")
                .attr("y", "50%")
                .attr("text-anchor", "middle")
                .attr("fill", "#e0e0e0")
                .attr("font-family", "serif")
                .attr("font-size", "20px")
                .text("Loading...");
            var svg = this._svg;
            // Add the zoom transformation <g> element
            this._zoom_node = this._svg.append('svg:g').attr('transform', 'scale(1)');
            this._zoom_scale = 1.0;
            this._zoom_xlat = [0.0, 0.0];
            // Create the container <g> element that will contain our scene
            this.world = this._zoom_node.append('g');
            // Add the image element to the container
            this._image = this.world.append("image")
                .attr("x", 0)
                .attr("y", 0);
            // Flag that indicates if the mouse pointer is within the tool area
            this._mouse_within = false;
            this._last_mouse_pos = null;
            // Create the root view
            this.root_view = new labelling_tool.RootLabelView(initial_model, this.root_view_listener, this._entity_event_listener, this, this.world);
            //
            // Set up event handlers
            //
            // Click
            this.world.on("click", function () {
                self.notifyStopwatchChanges();
                var click_event = d3.event;
                if (click_event.button === 0) {
                    // Left click; send to tool
                    if (!click_event.altKey) {
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_left_click(self.get_mouse_pos_world_space(), d3.event);
                        }
                        click_event.stopPropagation();
                    }
                }
            });
            // Button press
            this.world.on("mousedown", function () {
                self.notifyStopwatchChanges();
                var button_event = d3.event;
                if (button_event.button === 0) {
                    // Left button down
                    if (!button_event.altKey) {
                        self._button_down = true;
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_button_down(self.get_mouse_pos_world_space(), d3.event);
                        }
                        button_event.stopPropagation();
                    }
                }
                else if (button_event.button === 2) {
                    // Right click; on_cancel current tool
                    if (_this._current_tool !== null) {
                        var handled = _this._current_tool.on_cancel(self.get_mouse_pos_world_space());
                        if (handled) {
                            button_event.stopPropagation();
                        }
                    }
                }
            });
            // Button press
            this.world.on("mouseup", function () {
                self.notifyStopwatchChanges();
                var button_event = d3.event;
                if (button_event.button === 0) {
                    // Left buton up
                    if (!button_event.altKey) {
                        self._button_down = false;
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_button_up(self.get_mouse_pos_world_space(), d3.event);
                        }
                        button_event.stopPropagation();
                    }
                }
            });
            // Mouse on_move
            this.world.on("mousemove", function () {
                self.notifyStopwatchChanges();
                var move_event = d3.event;
                self._last_mouse_pos = self.get_mouse_pos_world_space();
                if (self._button_down) {
                    if (_this._current_tool !== null) {
                        _this._current_tool.on_drag(self._last_mouse_pos);
                    }
                    move_event.stopPropagation();
                }
                else {
                    if (!self._mouse_within) {
                        self._init_key_handlers();
                        // Entered tool area; invoke tool.on_switch_in()
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_switch_in(self._last_mouse_pos);
                        }
                        self._mouse_within = true;
                    }
                    else {
                        // Send mouse on_move event to tool
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_move(self._last_mouse_pos);
                        }
                    }
                }
            });
            // Mouse wheel
            this.world.on("mousewheel", function () {
                self.notifyStopwatchChanges();
                var wheel_event = d3.event;
                var handled = false;
                self._last_mouse_pos = self.get_mouse_pos_world_space();
                if (wheel_event.ctrlKey || wheel_event.shiftKey || wheel_event.altKey) {
                    if (_this._current_tool !== null) {
                        handled = _this._current_tool.on_wheel(self._last_mouse_pos, wheel_event.wheelDeltaX, wheel_event.wheelDeltaY);
                    }
                }
                if (handled) {
                    wheel_event.stopPropagation();
                }
            });
            var on_mouse_out = function (pos, width, height) {
                self.notifyStopwatchChanges();
                var mouse_event = d3.event;
                if (self._mouse_within) {
                    if (pos.x < 0.0 || pos.x > width || pos.y < 0.0 || pos.y > height) {
                        // The pointer is outside the bounds of the tool, as opposed to entering another element within the bounds of the tool, e.g. a polygon
                        // invoke tool.on_switch_out()
                        var handled = false;
                        if (_this._current_tool !== null) {
                            _this._current_tool.on_switch_out(self.get_mouse_pos_world_space());
                            handled = true;
                        }
                        if (handled) {
                            mouse_event.stopPropagation();
                        }
                        self._mouse_within = false;
                        self._last_mouse_pos = null;
                        self._shutdown_key_handlers();
                    }
                }
            };
            // Mouse leave
            this._svg.on("mouseout", function () {
                on_mouse_out(_this.get_mouse_pos_screen_space(), _this._labelling_area_width, _this._tool_height);
            });
            // Global key handler
            if (!LabellingTool._global_key_handler_connected) {
                d3.select("body").on("keydown", function () {
                    self.notifyStopwatchChanges();
                    if (LabellingTool._global_key_handler !== null) {
                        var key_event = d3.event;
                        var handled = LabellingTool._global_key_handler(key_event);
                        if (handled) {
                            key_event.stopPropagation();
                        }
                    }
                });
                LabellingTool._global_key_handler_connected = true;
            }
            // Create entities for the pre-existing labels
            if (initial_image_index < this._images.length) {
                this.loadImage(this._images[initial_image_index]);
            }
        }
        ;
        LabellingTool.prototype.on_key_down = function (event) {
            var handled = false;
            if (event.keyCode === 186) {
                if (this.label_visibility === labelling_tool.LabelVisibility.HIDDEN) {
                    this.set_label_visibility(labelling_tool.LabelVisibility.FULL);
                    this.label_vis_full_radio[0].checked = true;
                }
                else if (this.label_visibility === labelling_tool.LabelVisibility.FAINT) {
                    this.set_label_visibility(labelling_tool.LabelVisibility.HIDDEN);
                    this.label_vis_hidden_radio[0].checked = true;
                }
                else if (this.label_visibility === labelling_tool.LabelVisibility.FULL) {
                    this.set_label_visibility(labelling_tool.LabelVisibility.FAINT);
                    this.label_vis_faint_radio[0].checked = true;
                }
                else {
                    throw "Unknown label visibility " + this.label_visibility;
                }
                handled = true;
            }
            return handled;
        };
        ;
        LabellingTool.prototype._image_id_to_index = function (image_id) {
            for (var i = 0; i < this._images.length; i++) {
                if (this._images[i].image_id === image_id) {
                    return i;
                }
            }
            console.log("Image ID " + image_id + " not found");
            return 0;
        };
        ;
        LabellingTool.prototype._update_image_index_input_by_id = function (image_id) {
            var image_index = this._image_id_to_index(image_id);
            this._image_index_input.val((image_index + 1).toString());
        };
        ;
        LabellingTool.prototype._get_current_image_id = function () {
            return this.root_view.get_current_image_id();
        };
        ;
        LabellingTool.prototype.loadImageUrl = function (url) {
            var self = this;
            var img = new Image();
            var onload = function () {
                self._notify_image_loaded();
            };
            var onerror = function () {
                self._notify_image_error();
            };
            img.addEventListener('load', onload, false);
            img.addEventListener('error', onerror, false);
            img.src = url;
            return img;
        };
        LabellingTool.prototype.loadImage = function (image) {
            var self = this;
            // Update the image SVG element if the image URL is available
            if (image.img_url !== null) {
                var img = self.loadImageUrl(image.img_url);
                this._image.attr("width", image.width + 'px');
                this._image.attr("height", image.height + 'px');
                this._image.attr('xlink:href', img.src);
                this._image_width = image.width;
                this._image_height = image.height;
                this._image_initialised = true;
            }
            else {
                this._image_initialised = false;
            }
            this.root_view.set_model({
                image_id: "",
                complete: false,
                timeElapsed: 0.0,
                state: 'editable',
                labels: []
            });
            this._resetStopwatch();
            this._complete_checkbox[0].checked = false;
            this._image_index_input.val("");
            this.set_current_tool(null);
            this._requestLabelsCallback(image.image_id);
            this._image_loaded = false;
            this._labels_loaded = false;
            this._show_loading_notification();
        };
        LabellingTool.prototype.loadLabels = function (label_header, image) {
            var self = this;
            if (!this._image_initialised) {
                if (image !== null && image !== undefined) {
                    var img = self.loadImageUrl(image.img_url);
                    this._image.attr("width", image.width + 'px');
                    this._image.attr("height", image.height + 'px');
                    this._image.attr('xlink:href', img.src);
                    this._image_width = image.width;
                    this._image_height = image.height;
                    this._image_initialised = true;
                }
                else {
                    console.log("Labelling tool: Image URL was unavailable to loadImage and has not been " +
                        "provided by loadLabels");
                }
            }
            // Update the image SVG element
            this.root_view.set_model(label_header);
            this._resetStopwatch();
            this._update_image_index_input_by_id(this.root_view.model.image_id);
            if (this.root_view.model.state === 'locked') {
                this.lockLabels();
            }
            else {
                this.unlockLabels();
            }
            this._complete_checkbox[0].checked = this.root_view.model.complete;
            this.set_current_tool(new labelling_tool.SelectEntityTool(this.root_view));
            this._labels_loaded = true;
            this._hide_loading_notification_if_ready();
        };
        ;
        LabellingTool.prototype._notify_image_loaded = function () {
            this._image_loaded = true;
            this._hide_loading_notification_if_ready();
        };
        LabellingTool.prototype._notify_image_error = function () {
            var src = this._image.attr('xlink:href');
            console.log("Error loading image " + src);
            this._show_loading_notification();
            this._loading_notification_text.text("Error loading " + src);
        };
        LabellingTool.prototype._show_loading_notification = function () {
            this._svg.attr("style", "display: none");
            this._loading_notification.attr("style", "");
            this._loading_notification_text.text("Loading...");
        };
        LabellingTool.prototype._hide_loading_notification_if_ready = function () {
            if (this._image_loaded && this._labels_loaded) {
                this._svg.attr("style", "");
                this._loading_notification.attr("style", "display: none");
            }
        };
        LabellingTool.prototype.goToImageById = function (image_id) {
            if (image_id !== null && image_id !== undefined) {
                // Convert to string in case we go something else
                image_id = image_id.toString();
                for (var i = 0; i < this._images.length; i++) {
                    if (this._images[i].image_id === image_id) {
                        this.loadImage(this._images[i]);
                    }
                }
            }
        };
        LabellingTool.prototype.notifyLabelUpdateResponse = function (msg) {
            if (msg.error === undefined) {
                // All good
            }
            else if (msg.error === 'locked') {
                // Lock controls
                this.lockLabels();
            }
        };
        LabellingTool.prototype.notifyStopwatchChanges = function () {
            var self = this;
            var current = new Date().getTime();
            // Start the stopwatch if its not going
            if (this._stopwatchStart === null) {
                this._stopwatchStart = current;
            }
            this._stopwatchCurrent = current;
            if (this._stopwatchHandle !== null) {
                clearTimeout(this._stopwatchHandle);
                this._stopwatchHandle = null;
            }
            this._stopwatchHandle = setTimeout(function () {
                self._onStopwatchInactivity();
            }, this._config.settings.inactivityTimeoutMS);
        };
        LabellingTool.prototype._onStopwatchInactivity = function () {
            if (this._stopwatchHandle !== null) {
                clearTimeout(this._stopwatchHandle);
                this._stopwatchHandle = null;
            }
            var elapsed = this._stopwatchCurrent - this._stopwatchStart;
            this._stopwatchStart = this._stopwatchCurrent = null;
            this._notifyStopwatchElapsed(elapsed);
        };
        LabellingTool.prototype._resetStopwatch = function () {
            this._stopwatchStart = this._stopwatchCurrent = null;
            if (this._stopwatchHandle !== null) {
                clearTimeout(this._stopwatchHandle);
                this._stopwatchHandle = null;
            }
        };
        LabellingTool.prototype._notifyStopwatchElapsed = function (elapsed) {
            var t = this.root_view.model.timeElapsed;
            if (t === undefined || t === null) {
                t = 0.0;
            }
            t += (elapsed * 0.001);
            this.root_view.model.timeElapsed = t;
        };
        LabellingTool.prototype._commitStopwatch = function () {
            if (this._stopwatchStart !== null) {
                var current = new Date().getTime();
                var elapsed = current - this._stopwatchStart;
                this._notifyStopwatchElapsed(elapsed);
                this._stopwatchStart = this._stopwatchCurrent = current;
            }
        };
        LabellingTool.prototype.lockLabels = function () {
            this._lockableControls.attr('disabled', 'disable');
            this._lockNotification.removeAttr('style');
            this.set_current_tool(null);
        };
        LabellingTool.prototype.unlockLabels = function () {
            this._lockableControls.removeAttr('disabled');
            this._lockNotification.attr('style', 'display: none;');
            this.set_current_tool(new labelling_tool.SelectEntityTool(this.root_view));
        };
        /*
        Get colour for a given label class
         */
        LabellingTool.prototype.index_for_label_class = function (label_class) {
            if (label_class != null) {
                for (var i = 0; i < this.label_classes.length; i++) {
                    var cls = this.label_classes[i];
                    if (cls.name === label_class) {
                        return i;
                    }
                }
            }
            // Default
            return -1;
        };
        ;
        LabellingTool.prototype.colour_for_label_class = function (label_class) {
            var index = this.index_for_label_class(label_class);
            if (index !== -1) {
                return this.label_classes[index].colour;
            }
            else {
                // Default
                return labelling_tool.Colour4.BLACK;
            }
        };
        ;
        LabellingTool.prototype._update_label_class_menu = function (label_class) {
            if (label_class === null) {
                label_class = '__unclassified';
            }
            this._label_class_selector_menu.children('option').each(function () {
                this.selected = (this.value == label_class);
            });
        };
        ;
        LabellingTool.prototype._update_label_class_menu_from_views = function (selection) {
            if (selection.length === 1) {
                this._update_label_class_menu(selection[0].model.label_class);
            }
            else {
                this._update_label_class_menu(null);
            }
        };
        ;
        /*
        Set label visibility
         */
        LabellingTool.prototype.set_label_visibility = function (visibility) {
            this.label_visibility = visibility;
            this.root_view.set_label_visibility(visibility);
        };
        /*
        Set the current tool; switch the old one out and a new one in
         */
        LabellingTool.prototype.set_current_tool = function (tool) {
            if (this._current_tool !== null) {
                if (this._mouse_within) {
                    this._current_tool.on_switch_out(this._last_mouse_pos);
                }
                this._current_tool.on_shutdown();
            }
            this._current_tool = tool;
            if (this._current_tool !== null) {
                this._current_tool.on_init();
                if (this._mouse_within) {
                    this._current_tool.on_switch_in(this._last_mouse_pos);
                }
            }
        };
        ;
        LabellingTool.prototype.freeze = function () {
            this.frozen = true;
        };
        LabellingTool.prototype.thaw = function () {
            this.frozen = false;
        };
        LabellingTool.prototype.queue_push_label_data = function () {
            var _this = this;
            if (!this.frozen) {
                if (this._pushDataTimeout === null) {
                    this._pushDataTimeout = setTimeout(function () {
                        _this._pushDataTimeout = null;
                        _this._sendLabelHeaderFn(_this.root_view.model);
                    }, 0);
                }
            }
        };
        ;
        // Function for getting the current mouse position
        LabellingTool.prototype.get_mouse_pos_world_space = function () {
            var pos_screen = d3.mouse(this._svg[0][0]);
            return { x: (pos_screen[0] - this._zoom_xlat[0]) / this._zoom_scale,
                y: (pos_screen[1] - this._zoom_xlat[1]) / this._zoom_scale };
        };
        ;
        LabellingTool.prototype.get_mouse_pos_screen_space = function () {
            var pos = d3.mouse(this._svg[0][0]);
            return { x: pos[0], y: pos[1] };
        };
        ;
        LabellingTool.prototype._init_key_handlers = function () {
            var self = this;
            var on_key_down = function (event) {
                return self._overall_on_key_down(event);
            };
            LabellingTool._global_key_handler = on_key_down;
        };
        ;
        LabellingTool.prototype._shutdown_key_handlers = function () {
            LabellingTool._global_key_handler = null;
        };
        ;
        LabellingTool.prototype._overall_on_key_down = function (event) {
            if (this._mouse_within) {
                var handled = false;
                if (this._current_tool !== null) {
                    handled = this._current_tool.on_key_down(event);
                }
                if (!handled) {
                    handled = this.on_key_down(event);
                }
                return handled;
            }
            else {
                return false;
            }
        };
        return LabellingTool;
    }());
    labelling_tool.LabellingTool = LabellingTool;
})(labelling_tool || (labelling_tool = {}));
