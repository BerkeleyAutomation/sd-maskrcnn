import os
from ast import literal_eval

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

def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)