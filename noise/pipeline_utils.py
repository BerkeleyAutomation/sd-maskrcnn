import os
import model as modellib

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
