from importlib import import_module
import sys
import os.path as osp


def load_config(config_dir: str) -> dict:
    assert config_dir.endswith('.py')
    config_path, config_filename = osp.split(config_dir)
    temp_module_name = osp.splitext(config_filename)[0]
    sys.path.insert(0, config_path)
    mod = import_module(temp_module_name)
    sys.path.pop(0)
    cfg_dict = {
        key: value
        for key, value in mod.__dict__.items() if not key.startswith('__')
    }
    del sys.modules[temp_module_name]
    return cfg_dict
