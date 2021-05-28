from os import path
from munch import Munch
import inspect
import yaml




def load_yaml(yaml_filename):
    with open(yaml_filename, 'r') as stream:
        try:
            return Munch.fromDict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def rel_to_abs_path(relative_path):
    """convert caller's relative path to absolute path"""
    callers_path = inspect.stack()[1].filename
    return path.normpath(path.join(path.dirname(path.abspath(callers_path)), relative_path))


def get_abs_module_path():
    return rel_to_abs_path(".")
