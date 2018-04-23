import os
import model as modellib

def mkdir_if_missing(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print("Something went wrong in mkdir_if_missing. "
                  "Probably some other process created the directory already.")


def get_model(config, clutter_config):
    model_path = config["model_path"]
    mode = config["model_mode"]
    model_dir, model_name = os.path.split(model_path)
    model = modellib.MaskRCNN(mode=mode, config=clutter_config,
                              model_dir=model_dir)
    return model
