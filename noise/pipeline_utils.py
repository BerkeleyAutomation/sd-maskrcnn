import os
import model as modellib
from ast import literal_eval

def mkdir_if_missing(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print("Something went wrong in mkdir_if_missing. "
                  "Probably some other process created the directory already.")


def get_model_and_path(config, clutter_config):
    mode = config["model_mode"]
    model_folder = config["model_weights_folder"]
    base_path = config["base_path"]
    model_path = os.path.join(base_path, model_folder)
    print(model_path)
    mkdir_if_missing(model_path)
    model = modellib.MaskRCNN(mode=mode, config=clutter_config,
                              model_dir=model_path)
    return model, model_path


def get_conf_dict(config_obj):
    # create a dictionary of the proper arguments, including
    # the requested task
    task = config_obj.get("GENERAL", "task").upper()
    task = literal_eval(task)
    conf_dict = dict(config_obj.items(task))

    # return a type-sensitive version of the dictionary;
    # prevents further need to cast from string to other types
    out = {}
    for key, value in conf_dict.items():
        out[key] = literal_eval(value)

    out["task"] = task

    return out
